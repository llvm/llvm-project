//=== unittests/CodeGen/IRMatchers.h - Match on the LLVM IR -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides a simple mechanism for performing search operations over
/// IR including metadata and types. It allows writing complex search patterns
/// using understandable syntax. For instance, the code:
///
/// \code
///       const BasicBlock *BB = ...
///       const Instruction *I = match(BB,
///           MInstruction(Instruction::Store,
///               MConstInt(4, 8),
///               MMTuple(
///                   MMTuple(
///                       MMString("omnipotent char"),
///                       MMTuple(
///                           MMString("Simple C/C++ TBAA")),
///                       MConstInt(0, 64)),
///                   MSameAs(0),
///                   MConstInt(0))));
/// \endcode
///
/// searches the basic block BB for the 'store' instruction, first argument of
/// which is 'i8 4', and the attached metadata has an item described by the
/// given tree.
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_CODEGEN_IRMATCHERS_H
#define CLANG_UNITTESTS_CODEGEN_IRMATCHERS_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Value.h"

namespace llvm {

/// Keeps information about pending match queries.
///
/// This class stores state of all unfinished match actions. It allows to
/// use queries like "this operand is the same as n-th operand", which are
/// hard to implement otherwise.
///
class MatcherContext {
public:

  /// Describes pending match query.
  ///
  /// The query is represented by the current entity being investigated (type,
  /// value or metadata). If the entity is a member of a list (like arguments),
  /// the query also keeps the entity number in that list.
  ///
  class Query {
    PointerUnion3<const Value *, const Metadata *, const Type *> Entity;
    unsigned OperandNo;

  public:
    Query(const Value *V, unsigned N) : Entity(V), OperandNo(N) {}
    Query(const Metadata *M, unsigned N) : Entity(M), OperandNo(N) {}
    Query(const Type *T, unsigned N) : Entity(T), OperandNo(N) {}

    template<typename T>
    const T *get() const {
      return Entity.dyn_cast<const T *>();
    }

    unsigned getOperandNo() const { return OperandNo; }
  };

  template<typename T>
  void push(const T *V, unsigned N = ~0) {
    MatchStack.push_back(Query(V, N));
  }

  void pop() { MatchStack.pop_back(); }

  template<typename T>
  const T *top() const { return MatchStack.back().get<T>(); }

  size_t size() const { return MatchStack.size(); }

  unsigned getOperandNo() const { return MatchStack.back().getOperandNo(); }

  /// Returns match query at the given offset from the top of queries.
  ///
  /// Offset 0 corresponds to the topmost query.
  ///
  const Query &getQuery(unsigned Offset) const {
    assert(MatchStack.size() > Offset);
    return MatchStack[MatchStack.size() - 1 - Offset];
  }

private:
  SmallVector<Query, 8> MatchStack;
};


/// Base of all matcher classes.
///
class Matcher {
public:
  virtual ~Matcher() {}

  /// Returns true if the entity on the top of the specified context satisfies
  /// the matcher condition.
  ///
  virtual bool match(MatcherContext &MC) = 0;
};


/// Base class of matchers that test particular entity.
///
template<typename T>
class EntityMatcher : public Matcher {
public:
  bool match(MatcherContext &MC) override {
    if (auto V = MC.top<T>())
      return matchEntity(*V, MC);
    return false;
  }
  virtual bool matchEntity(const T &M, MatcherContext &C) = 0;
};


/// Matcher that matches any entity of the specified kind.
///
template<typename T>
class AnyMatcher : public EntityMatcher<T> {
public:
  bool matchEntity(const T &M, MatcherContext &C) override { return true; }
};


/// Matcher that tests if the current entity satisfies the specified
/// condition.
///
template<typename T>
class CondMatcher : public EntityMatcher<T> {
  std::function<bool(const T &)> Condition;
public:
  CondMatcher(std::function<bool(const T &)> C) : Condition(C) {}
  bool matchEntity(const T &V, MatcherContext &C) override {
    return Condition(V);
  }
};


/// Matcher that save pointer to the entity that satisfies condition of the
// specified matcher.
///
template<typename T>
class SavingMatcher : public EntityMatcher<T> {
  const T *&Var;
  std::shared_ptr<Matcher> Next;
public:
  SavingMatcher(const T *&V, std::shared_ptr<Matcher> N) : Var(V), Next(N) {}
  bool matchEntity(const T &V, MatcherContext &C) override {
    bool Result = Next->match(C);
    if (Result)
      Var = &V;
    return Result;
  }
};


/// Matcher that checks that the entity is identical to another entity in the
/// same container.
///
class SameAsMatcher : public Matcher {
  unsigned OpNo;
public:
  SameAsMatcher(unsigned N) : OpNo(N) {}
  bool match(MatcherContext &C) override {
    if (C.getOperandNo() != ~0U) {
      // Handle all known containers here.
      const MatcherContext::Query &StackRec = C.getQuery(1);
      if (const Metadata *MR = StackRec.get<Metadata>()) {
        if (const auto *MT = dyn_cast<MDTuple>(MR)) {
          if (OpNo < MT->getNumOperands())
            return C.top<Metadata>() == MT->getOperand(OpNo).get();
          return false;
        }
        llvm_unreachable("Unknown metadata container");
      }
      if (const Value *VR = StackRec.get<Value>()) {
        if (const auto *Insn = dyn_cast<Instruction>(VR)) {
          if (OpNo < Insn->getNumOperands())
            return C.top<Value>() == Insn->getOperand(OpNo);
          return false;
        }
        llvm_unreachable("Unknown value container");
      }
      llvm_unreachable("Unknown type container");
    }
    return false;
  }
};


/// Matcher that tests if the entity is a constant integer.
///
class ConstantIntMatcher : public Matcher {
  uint64_t IntValue;
  unsigned Width;
public:
  ConstantIntMatcher(uint64_t V, unsigned W = 0) : IntValue(V), Width(W) {}
  bool match(MatcherContext &Ctx) override {
    if (const Value *V = Ctx.top<Value>()) {
      if (const auto *CI = dyn_cast<ConstantInt>(V))
        return (Width == 0 || CI->getBitWidth() == Width) &&
               CI->getLimitedValue() == IntValue;
    }
    if (const Metadata *M = Ctx.top<Metadata>()) {
      if (const auto *MT = dyn_cast<ValueAsMetadata>(M))
        if (const auto *C = dyn_cast<ConstantInt>(MT->getValue()))
          return (Width == 0 || C->getBitWidth() == Width) &&
                 C->getLimitedValue() == IntValue;
    }
    return false;
  }
};


/// Value matcher tuned to test instructions.
///
class InstructionMatcher : public EntityMatcher<Value> {
  SmallVector<std::shared_ptr<Matcher>, 8> OperandMatchers;
  std::shared_ptr<EntityMatcher<Metadata>> MetaMatcher = nullptr;
  unsigned Code;
public:
  InstructionMatcher(unsigned C) : Code(C) {}

  void push(std::shared_ptr<EntityMatcher<Metadata>> M) {
    assert(!MetaMatcher && "Only one metadata matcher may be specified");
    MetaMatcher = M;
  }
  void push(std::shared_ptr<Matcher> V) { OperandMatchers.push_back(V); }
  template<typename... Args>
  void push(std::shared_ptr<Matcher> V, Args... A) {
    push(V);
    push(A...);
  }

  virtual bool matchInstruction(const Instruction &I) {
    return I.getOpcode() == Code;
  }

  bool matchEntity(const Value &V, MatcherContext &C) override {
    if (const auto *I = dyn_cast<Instruction>(&V)) {
      if (!matchInstruction(*I))
        return false;
      if (OperandMatchers.size() > I->getNumOperands())
        return false;
      for (unsigned N = 0, E = OperandMatchers.size(); N != E; ++N) {
        C.push(I->getOperand(N), N);
        if (!OperandMatchers[N]->match(C)) {
          C.pop();
          return false;
        }
        C.pop();
      }
      if (MetaMatcher) {
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        I->getAllMetadata(MDs);
        bool Found = false;
        for (auto Item : MDs) {
          C.push(Item.second);
          if (MetaMatcher->match(C)) {
            Found = true;
            C.pop();
            break;
          }
          C.pop();
        }
        return Found;
      }
      return true;
    }
    return false;
  }
};


/// Matcher that tests type of the current value using the specified
/// type matcher.
///
class ValueTypeMatcher : public EntityMatcher<Value> {
  std::shared_ptr<EntityMatcher<Type>> TyM;
public:
  ValueTypeMatcher(std::shared_ptr<EntityMatcher<Type>> T) : TyM(T) {}
  ValueTypeMatcher(const Type *T)
    : TyM(new CondMatcher<Type>([T](const Type &Ty) -> bool {
                                  return &Ty == T;
                                })) {}
  bool matchEntity(const Value &V, MatcherContext &Ctx) override {
    Type *Ty = V.getType();
    Ctx.push(Ty);
    bool Res = TyM->match(Ctx);
    Ctx.pop();
    return Res;
  }
};


/// Matcher that matches string metadata.
///
class NameMetaMatcher : public EntityMatcher<Metadata> {
  StringRef Name;
public:
  NameMetaMatcher(StringRef N) : Name(N) {}
  bool matchEntity(const Metadata &M, MatcherContext &C) override {
    if (auto *MDS = dyn_cast<MDString>(&M))
      return MDS->getString().equals(Name);
    return false;
  }
};


/// Matcher that matches metadata tuples.
///
class MTupleMatcher : public EntityMatcher<Metadata> {
  SmallVector<std::shared_ptr<Matcher>, 4> Operands;
public:
  void push(std::shared_ptr<Matcher> M) { Operands.push_back(M); }
  template<typename... Args>
  void push(std::shared_ptr<Matcher> M, Args... A) {
    push(M);
    push(A...);
  }
  bool matchEntity(const Metadata &M, MatcherContext &C) override {
    if (const auto *MT = dyn_cast<MDTuple>(&M)) {
      if (MT->getNumOperands() != Operands.size())
        return false;
      for (unsigned I = 0, E = MT->getNumOperands(); I != E; ++I) {
        const MDOperand &Op = MT->getOperand(I);
        C.push(Op.get(), I);
        if (!Operands[I]->match(C)) {
          C.pop();
          return false;
        }
        C.pop();
      }
      return true;
    }
    return false;
  }
};


// Helper function used to construct matchers.

std::shared_ptr<Matcher> MSameAs(unsigned N) {
  return std::shared_ptr<Matcher>(new SameAsMatcher(N));
}

template<typename... T>
std::shared_ptr<InstructionMatcher> MInstruction(unsigned C, T... Args) {
  auto Result = new InstructionMatcher(C);
  Result->push(Args...);
  return std::shared_ptr<InstructionMatcher>(Result);
}

std::shared_ptr<Matcher> MConstInt(uint64_t V, unsigned W = 0) {
  return std::shared_ptr<Matcher>(new ConstantIntMatcher(V, W));
}

std::shared_ptr<EntityMatcher<Value>>
 MValType(std::shared_ptr<EntityMatcher<Type>> T) {
  return std::shared_ptr<EntityMatcher<Value>>(new ValueTypeMatcher(T));
}

std::shared_ptr<EntityMatcher<Value>> MValType(const Type *T) {
  return std::shared_ptr<EntityMatcher<Value>>(new ValueTypeMatcher(T));
}

std::shared_ptr<EntityMatcher<Type>>
MType(std::function<bool(const Type &)> C) {
  return std::shared_ptr<EntityMatcher<Type>>(new CondMatcher<Type>(C));
}

std::shared_ptr<EntityMatcher<Metadata>> MMAny() {
  return std::shared_ptr<EntityMatcher<Metadata>>(new AnyMatcher<Metadata>);
}

std::shared_ptr<EntityMatcher<Metadata>>
MMSave(const Metadata *&V, std::shared_ptr<EntityMatcher<Metadata>> M) {
  return std::shared_ptr<EntityMatcher<Metadata>>(
      new SavingMatcher<Metadata>(V, M));
}

std::shared_ptr<EntityMatcher<Metadata>>
MMString(const char *Name) {
  return std::shared_ptr<EntityMatcher<Metadata>>(new NameMetaMatcher(Name));
}

template<typename... T>
std::shared_ptr<EntityMatcher<Metadata>> MMTuple(T... Args) {
  auto Res = new MTupleMatcher();
  Res->push(Args...);
  return std::shared_ptr<EntityMatcher<Metadata>>(Res);
}


/// Looks for the instruction that satisfies condition of the specified
/// matcher inside the given basic block.
/// \returns Pointer to the found instruction or nullptr if such instruction
///          was not found.
///
const Instruction *match(const BasicBlock *BB, std::shared_ptr<Matcher> M) {
  MatcherContext MC;
  for (const auto &I : *BB) {
    MC.push(&I);
    if (M->match(MC))
      return &I;
    MC.pop();
  }
  assert(MC.size() == 0);
  return nullptr;
}


/// Looks for the instruction that satisfies condition of the specified
/// matcher starting from the specified instruction inside the same basic block.
///
/// The given instruction is not checked.
///
const Instruction *matchNext(const Instruction *I, std::shared_ptr<Matcher> M) {
  if (!I)
    return nullptr;
  MatcherContext MC;
  const BasicBlock *BB = I->getParent();
  if (!BB)
    return nullptr;
  for (auto P = ++BasicBlock::const_iterator(I), E = BB->end(); P != E; ++P) {
    MC.push(&*P);
    if (M->match(MC))
      return &*P;
    MC.pop();
  }
  assert(MC.size() == 0);
  return nullptr;
}

}
#endif
