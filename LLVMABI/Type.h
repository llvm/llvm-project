#ifndef TYPE_H
#define TYPE_H


namespace ABI{
class Type {
  public:
  enum TypeClass { Builtin };
  explicit Type(TypeClass TC) : Tc(TC) {} 

    TypeClass getTypeClass() const { return Tc; }
    virtual ~Type() = default;

  private:
  //save the type class
    TypeClass Tc;
  }

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
      // ABIType(Kind K = Integer);
      Kind getKind() const { return kind;}

      bool isInteger() const {
        return getKind() == Kind::Integer || getKind() == Kind::SignedInteger || getKind() == Kind::UnsignedInteger || getKind() == Kind::Int128;
      }

      bool isSignedInteger() const {
        return getKind() == Kind::SignedInteger;
      }

      bool isUnsignedInteger() const {
        return getKind() == Kind::UnsignedInteger;
      }

      bool isFloatingPoint() const {
        return getKind() == Kind::FloatingPoint;
      }

  }

}

#endif