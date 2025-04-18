#ifndef TYPE_H
#define TYPE_H

#include "llvm/include/llvm/ADT/PointerIntPair.h"
#include <string>
#include <vector>

using namespace std;

  
namespace ABI{
  class Type {
    public:
    enum TypeClass { Builtin, Record };
    
    Type(TypeClass TC) : Tc(TC) {}

    // getter method
    TypeClass getTypeClass() const { return Tc; }

    virtual ~Type() = default;
    bool isBuiltinType() const { return getTypeClass() == Builtin; }
    bool isAggregateType() const { return getTypeClass() == Record; }

    static bool classof(const Type *T) { return true; }

    // debug info
    virtual void dump() const;

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
      Int128,
      UInt128,
      LongLong,
      Float,
      Double,
      Float16,
      BFloat16,
      Float128,
      LongDouble
    };

  private:
    Kind kind;
    uint64_t Size;
    uint64_t Alignment;

  public:
    ABIBuiltinType(Kind K) : Type(Builtin), kind(K) {}

    Kind getKind() const { return kind; }

    bool isInteger() const {
      return getKind() == Kind::Integer || getKind() == Kind::Int128;
    }
    bool isFloatingPoint() const { return getKind() == Kind::Float; }

    static bool classof(const Type T) {
      return T.getTypeClass() == Builtin;
    }

    void dump() const override;
};

class ABIRecordType : public Type {
  public:
    struct Field {
      std::string Name;
      const Type *FieldType;
      uint64_t OffsetInBits; // is this needed?

      Field(const std::string &N, const Type *T, uint64_t Offset = 0)
          : Name(N), FieldType(T), OffsetInBits(Offset) {}
    };

    using FieldList = std::vector<Field>;

    ABIRecordType(const std::string &Name, uint64_t AlignInBits = 0)
        : Type(Record), RecordName(Name), Alignment(AlignInBits) {}

    void addField(const std::string &Name, const Type *T, uint64_t Offset = 0) {
      Fields.emplace_back(Name, T,
                          Offset); // --> can I use soem other method for this?
    }

    const FieldList &getFields() const { return Fields; }

    uint64_t getAlignmentInBits() const { return Alignment; }
    void setAlignmentInBits(uint64_t Align) { Alignment = Align; }

    const std::string &getName() const { return RecordName; }

    static bool classof(const Type T) {
      return T.getTypeClass() == Record;
    }

    void dump() const;

  private:
    std::string RecordName;
    FieldList Fields;
    uint64_t Alignment;
};

} // namespace ABI

#endif

