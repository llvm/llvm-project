//===-- unittests/Evaluate/intrinsics.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/provenance.h"
#include "flang/Testing/testing.h"
#include "llvm/Support/raw_ostream.h"
#include <initializer_list>
#include <map>
#include <string>

namespace Fortran::evaluate {

class CookedStrings {
public:
  CookedStrings() {}
  explicit CookedStrings(const std::initializer_list<std::string> &ss) {
    for (const auto &s : ss) {
      Save(s);
    }
    Marshal();
  }
  void Save(const std::string &s) {
    offsets_[s] = cooked_.Put(s);
    cooked_.PutProvenance(allSources_.AddCompilerInsertion(s));
  }
  void Marshal() { cooked_.Marshal(allCookedSources_); }
  parser::CharBlock operator()(const std::string &s) {
    return {cooked_.AsCharBlock().begin() + offsets_[s], s.size()};
  }
  parser::ContextualMessages Messages(parser::Messages &buffer) {
    return parser::ContextualMessages{cooked_.AsCharBlock(), &buffer};
  }
  void Emit(llvm::raw_ostream &o, const parser::Messages &messages) {
    messages.Emit(o, allCookedSources_);
  }

private:
  parser::AllSources allSources_;
  parser::AllCookedSources allCookedSources_{allSources_};
  parser::CookedSource &cooked_{allCookedSources_.NewCookedSource()};
  std::map<std::string, std::size_t> offsets_;
};

template <typename A> auto Const(A &&x) -> Constant<TypeOf<A>> {
  const int kind{x.kind()};
  return Constant<TypeOf<A>>{kind, std::move(x)};
}

template <typename A> struct NamedArg {
  std::string keyword;
  A value;
};

template <typename A> static NamedArg<A> Named(std::string kw, A &&x) {
  return {kw, std::move(x)};
}

struct TestCall {
  TestCall(const common::IntrinsicTypeDefaultKinds &d,
      const IntrinsicProcTable &t, std::string n)
      : defaults{d}, table{t}, name{n} {}
  template <typename A> TestCall &Push(A &&x) {
    args.emplace_back(AsGenericExpr(std::move(x)));
    keywords.push_back("");
    return *this;
  }
  template <typename A> TestCall &Push(NamedArg<A> &&x) {
    args.emplace_back(AsGenericExpr(std::move(x.value)));
    keywords.push_back(x.keyword);
    strings.Save(x.keyword);
    return *this;
  }
  template <typename A, typename... As> TestCall &Push(A &&x, As &&...xs) {
    Push(std::move(x));
    return Push(std::move(xs)...);
  }
  void Marshal() {
    strings.Save(name);
    strings.Marshal();
    std::size_t j{0};
    for (auto &kw : keywords) {
      if (!kw.empty()) {
        args[j]->set_keyword(strings(kw));
      }
      ++j;
    }
  }
  void DoCall(std::optional<DynamicType> resultType = std::nullopt,
      int rank = 0, bool isElemental = false) {
    Marshal();
    parser::CharBlock fName{strings(name)};
    llvm::outs() << "function: " << fName.ToString();
    char sep{'('};
    for (const auto &a : args) {
      llvm::outs() << sep;
      sep = ',';
      a->AsFortran(llvm::outs());
    }
    if (sep == '(') {
      llvm::outs() << '(';
    }
    llvm::outs() << ')' << '\n';
    llvm::outs().flush();
    CallCharacteristics call{fName.ToString()};
    auto messages{strings.Messages(buffer)};
    TargetCharacteristics targetCharacteristics;
    common::LanguageFeatureControl languageFeatures;
    FoldingContext context{messages, defaults, table, targetCharacteristics,
        languageFeatures, tempNames};
    std::optional<SpecificCall> si{table.Probe(call, args, context)};
    if (resultType.has_value()) {
      TEST(si.has_value());
      TEST(messages.messages() && !messages.messages()->AnyFatalError());
      if (si) {
        const auto &proc{si->specificIntrinsic.characteristics.value()};
        const auto &fr{proc.functionResult};
        TEST(fr.has_value());
        if (fr) {
          const auto *ts{fr->GetTypeAndShape()};
          TEST(ts != nullptr);
          if (ts) {
            TEST(*resultType == ts->type());
            MATCH(rank, ts->Rank());
          }
        }
        MATCH(isElemental,
            proc.attrs.test(characteristics::Procedure::Attr::Elemental));
      }
    } else {
      TEST(!si.has_value());
      TEST((messages.messages() && messages.messages()->AnyFatalError()) ||
          name == "bad");
    }
    strings.Emit(llvm::outs(), buffer);
  }

  const common::IntrinsicTypeDefaultKinds &defaults;
  const IntrinsicProcTable &table;
  CookedStrings strings;
  parser::Messages buffer;
  ActualArguments args;
  std::string name;
  std::vector<std::string> keywords;
  std::set<std::string> tempNames;
};

void TestIntrinsics() {
  common::IntrinsicTypeDefaultKinds defaults;
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Integer));
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Real));
  IntrinsicProcTable table{IntrinsicProcTable::Configure(defaults)};
  table.Dump(llvm::outs());

  Type<TypeCategory::Integer> int1{1};
  Type<TypeCategory::Integer> int4{4};
  Type<TypeCategory::Integer> int8{8};
  Type<TypeCategory::Real> real4{4};
  Type<TypeCategory::Real> real8{8};
  Type<TypeCategory::Complex> complex4{4};
  Type<TypeCategory::Complex> complex8{8};

  TestCall{defaults, table, "bad"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(); // bad intrinsic name
  TestCall{defaults, table, "abs"}
      .Push(Named("a", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Named("bad", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "abs"}.DoCall(); // insufficient args
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(); // too many args
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .Push(Named("a", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall();
  TestCall{defaults, table, "abs"}
      .Push(Named("a", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall();
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(1)))
      .DoCall(int1.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(8)))
      .DoCall(int8.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)))
      .DoCall(real4.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(8)))
      .DoCall(real8.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(4)))
      .DoCall(real4.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(8)))
      .DoCall(real8.GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Character>>::Zero(1)))
      .DoCall();
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Type<TypeCategory::Logical>>::Zero(4)))
      .DoCall();

  // "Ext" in names for calls allowed as extensions
  TestCall maxCallR{defaults, table, "max"}, maxCallI{defaults, table, "min"},
      max0Call{defaults, table, "max0"}, max1Call{defaults, table, "max1"},
      amin0Call{defaults, table, "amin0"}, amin1Call{defaults, table, "amin1"},
      max0ExtCall{defaults, table, "max0"},
      amin1ExtCall{defaults, table, "amin1"};
  for (int j{0}; j < 10; ++j) {
    maxCallR.Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)));
    maxCallI.Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)));
    max0Call.Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)));
    max0ExtCall.Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)));
    max1Call.Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)));
    amin0Call.Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)));
    amin1ExtCall.Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)));
    amin1Call.Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)));
  }
  maxCallR.DoCall(real4.GetType());
  maxCallI.DoCall(int4.GetType());
  max0Call.DoCall(int4.GetType());
  max0ExtCall.DoCall(int4.GetType());
  max1Call.DoCall(int4.GetType());
  amin0Call.DoCall(real4.GetType());
  amin1Call.DoCall(real4.GetType());
  amin1ExtCall.DoCall(real4.GetType());

  TestCall{defaults, table, "conjg"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(4)))
      .DoCall(complex4.GetType());
  TestCall{defaults, table, "conjg"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(8)))
      .DoCall(complex8.GetType());
  TestCall{defaults, table, "dconjg"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(8)))
      .DoCall(complex8.GetType());

  TestCall{defaults, table, "float"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)))
      .DoCall();
  TestCall{defaults, table, "float"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(real4.GetType());
  TestCall{defaults, table, "idint"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall();
  TestCall{defaults, table, "idint"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(8)))
      .DoCall(int4.GetType());

  // Allowed as extensions
  TestCall{defaults, table, "float"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(8)))
      .DoCall(real4.GetType());
  TestCall{defaults, table, "idint"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)))
      .DoCall(int4.GetType());

  TestCall{defaults, table, "num_images"}.DoCall(int4.GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(1)))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(8)))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Named(
          "team_number", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall(int4.GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(); // too many args
  TestCall{defaults, table, "num_images"}
      .Push(Named("bad", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Character>>::Zero(1)))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Logical>>::Zero(4)))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(8)))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)))
      .DoCall(); // bad type

  // This test temporarily removed because it requires access to
  // the ISO_FORTRAN_ENV intrinsic module. This module should to
  // be loaded (somehow) and the following test reinstated.
  // TestCall{defaults, table, "team_number"}.DoCall(int4.GetType());

  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(); // too many args
  TestCall{defaults, table, "team_number"}
      .Push(Named("bad", Const(Scalar<Type<TypeCategory::Integer>>::Zero(4))))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Integer>>::Zero(4)))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Character>>::Zero(1)))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Logical>>::Zero(4)))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Complex>>::Zero(8)))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Type<TypeCategory::Real>>::Zero(4)))
      .DoCall(); // bad type

  // TODO: test other intrinsics

  // Test unrestricted specific to generic name mapping (table 16.2).
  TEST(table.GetGenericIntrinsicName("alog") == "log");
  TEST(table.GetGenericIntrinsicName("alog10") == "log10");
  TEST(table.GetGenericIntrinsicName("amod") == "mod");
  TEST(table.GetGenericIntrinsicName("cabs") == "abs");
  TEST(table.GetGenericIntrinsicName("ccos") == "cos");
  TEST(table.GetGenericIntrinsicName("cexp") == "exp");
  TEST(table.GetGenericIntrinsicName("clog") == "log");
  TEST(table.GetGenericIntrinsicName("csin") == "sin");
  TEST(table.GetGenericIntrinsicName("csqrt") == "sqrt");
  TEST(table.GetGenericIntrinsicName("dabs") == "abs");
  TEST(table.GetGenericIntrinsicName("dacos") == "acos");
  TEST(table.GetGenericIntrinsicName("dasin") == "asin");
  TEST(table.GetGenericIntrinsicName("datan") == "atan");
  TEST(table.GetGenericIntrinsicName("datan2") == "atan2");
  TEST(table.GetGenericIntrinsicName("dcos") == "cos");
  TEST(table.GetGenericIntrinsicName("dcosh") == "cosh");
  TEST(table.GetGenericIntrinsicName("ddim") == "dim");
  TEST(table.GetGenericIntrinsicName("derf") == "erf");
  TEST(table.GetGenericIntrinsicName("dexp") == "exp");
  TEST(table.GetGenericIntrinsicName("dint") == "aint");
  TEST(table.GetGenericIntrinsicName("dlog") == "log");
  TEST(table.GetGenericIntrinsicName("dlog10") == "log10");
  TEST(table.GetGenericIntrinsicName("dmod") == "mod");
  TEST(table.GetGenericIntrinsicName("dnint") == "anint");
  TEST(table.GetGenericIntrinsicName("dsign") == "sign");
  TEST(table.GetGenericIntrinsicName("dsin") == "sin");
  TEST(table.GetGenericIntrinsicName("dsinh") == "sinh");
  TEST(table.GetGenericIntrinsicName("dsqrt") == "sqrt");
  TEST(table.GetGenericIntrinsicName("dtan") == "tan");
  TEST(table.GetGenericIntrinsicName("dtanh") == "tanh");
  TEST(table.GetGenericIntrinsicName("iabs") == "abs");
  TEST(table.GetGenericIntrinsicName("idim") == "dim");
  TEST(table.GetGenericIntrinsicName("idnint") == "nint");
  TEST(table.GetGenericIntrinsicName("isign") == "sign");
  // Test a case where specific and generic name are the same.
  TEST(table.GetGenericIntrinsicName("acos") == "acos");
}
} // namespace Fortran::evaluate

int main() {
  Fortran::evaluate::TestIntrinsics();
  return testing::Complete();
}
