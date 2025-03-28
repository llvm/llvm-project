#ifndef TYPE_H
#define TYPE_H

#include "llvm/include/llvm/ADT/PointerIntPair.h"

namespace ABI{
class Type {
  public:
  enum TypeClass { Builtin };

  Type(TypeClass TC) : Tc(TC) {} 
  TypeClass getTypeClass() const { return Tc; }
  virtual ~Type() = default;

  private:
  //save the type class
    TypeClass Tc;
};

class ABIBuiltinType : public Type {
  public:
    enum Kind {
      Void,
      Bool,
      Integer,
      SignedInteger,
      UnsignedInteger,
      Int128,
      FloatingPoint
    };

  private:
  Kind kind; 

  public:
    ABIBuiltinType(Kind K)
        : Type(Builtin), kind(K){}

    Kind getKind() const { return kind;}

    bool isInteger() const { return getKind() == Kind::Integer || getKind() == Kind::SignedInteger || getKind() == Kind::UnsignedInteger || getKind() == Kind::Int128; }
    bool isSignedInteger() const { return getKind() == Kind::SignedInteger; }
    bool isUnsignedInteger() const { return getKind() == Kind::UnsignedInteger; }
    bool isFloatingPoint() const { return getKind() == Kind::FloatingPoint; }

};

class ABIQualifiers {
public:
  enum QualifierFlags {
    Const = 1 << 0,
    Volatile = 1 << 1,
    Restrict = 1 << 2,
  };

private:
  unsigned Quals;

public:
  ABIQualifiers() : Quals(0) {}
  ABIQualifiers(unsigned Q) : Quals(Q) {}

  bool hasConst() const { return Quals & Const; }
  bool hasVolatile() const { return Quals & Volatile; }
  bool hasRestrict() const { return Quals & Restrict; }

  unsigned getQualifiers() const { return Quals; }
};

class ABIQualType {
  // similar to how clang implements QualTypes. I don't think this is dependent on AST at all, just on support. Verify once.
  llvm::PointerIntPair<const Type *, 3, unsigned> Value; // 3 bits for qualifiers

public:
  ABIQualType() = default;
  ABIQualType(const Type *T, unsigned Quals = 0) : Value(T, Quals) {}

  const Type *getTypePtr() const { return Value.getPointer(); }

  bool isConstQualified() const { return Value.getInt() & ABIQualifiers::Const; }
  bool isVolatileQualified() const { return Value.getInt() & ABIQualifiers::Volatile; }
  bool isRestrictQualified() const { return Value.getInt() & ABIQualifiers::Restrict; }

  ABIQualifiers getQualifiers() const { return ABIQualifiers(Value.getInt()); }

  ABIQualType withConst() const { return ABIQualType(getTypePtr(), Value.getInt() | ABIQualifiers::Const); }
  ABIQualType withVolatile() const { return ABIQualType(getTypePtr(), Value.getInt() | ABIQualifiers::Volatile); }
  ABIQualType withRestrict() const { return ABIQualType(getTypePtr(), Value.getInt() | ABIQualifiers::Restrict); }
};

} // namespace ABI

#endif