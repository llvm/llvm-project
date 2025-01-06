// RUN: rm -rf %t
// RUN: split-file %s %t

// Build first header file
// RUN: echo "#define FIRST" >> %t/include/first.h
// RUN: cat %t/test.c        >> %t/include/first.h
// RUN: echo "#undef FIRST"  >> %t/include/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/include/second.h
// RUN: cat %t/test.c         >> %t/include/second.h
// RUN: echo "#undef SECOND"  >> %t/include/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/include/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/include/second.h -fblocks -fobjc-arc

// Run test
// RUN: %clang_cc1 -I%t/include -verify %t/test.c -fblocks -fobjc-arc \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Run tests for nested structs
// DEFINE: %{filename} = test-nested-struct.c
// DEFINE: %{macro_flag} = -DCASE1=1
// DEFINE: %{command} = %clang_cc1 -I%t/include -verify %t/%{filename} -fblocks -fobjc-arc \
// DEFINE:             -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache \
// DEFINE:             %{macro_flag} -emit-llvm -o %t/%{filename}.bc
// RUN: %{command}
// REDEFINE: %{macro_flag} = -DCASE2=1
// RUN: %{command}
// REDEFINE: %{macro_flag} = -DCASE3=1
// RUN: %{command}

// Run tests for anonymous nested structs and unions
// REDEFINE: %{filename} = test-anonymous.c
// REDEFINE: %{macro_flag} = -DCASE1=1
// RUN: %{command}
// REDEFINE: %{macro_flag} = -DCASE2=1
// RUN: %{command}
// REDEFINE: %{macro_flag} = -DCASE3=1
// RUN: %{command}

// Test that we don't accept different structs and unions with the same name
// from multiple modules but detect mismatches and provide actionable
// diagnostic.

//--- include/first-empty.h
//--- include/module.modulemap
module First {
  module Empty {
    header "first-empty.h"
  }
  module Hidden {
    header "first.h"
    header "first-nested-struct.h"
    header "first-anonymous.h"
    export *
  }
}
module Second {
  header "second.h"
  header "second-nested-struct.h"
  header "second-anonymous.h"
  export *
}

//--- test.c
#if !defined(FIRST) && !defined(SECOND)
# include "first-empty.h"
# include "second.h"
#endif

#if defined(FIRST)
struct CompareForwardDeclaration1;
struct CompareForwardDeclaration2 {};
#elif defined(SECOND)
struct CompareForwardDeclaration1 {};
struct CompareForwardDeclaration2;
#else
struct CompareForwardDeclaration1 *compareForwardDeclaration1;
struct CompareForwardDeclaration2 *compareForwardDeclaration2;
#endif

#if defined(FIRST)
struct CompareMatchingFields {
  int matchingFieldName;
};

struct CompareFieldPresence1 {
  int fieldPresence1;
};
struct CompareFieldPresence2 {};

struct CompareFieldName {
  int fieldNameA;
};

struct CompareFieldOrder {
  int fieldOrderX;
  int fieldOrderY;
};
#elif defined(SECOND)
struct CompareMatchingFields {
  int matchingFieldName;
};

struct CompareFieldPresence1 {
};
struct CompareFieldPresence2 {
  int fieldPresence2;
};

struct CompareFieldName {
  int fieldNameB;
};

struct CompareFieldOrder {
  int fieldOrderY;
  int fieldOrderX;
};
#else
struct CompareMatchingFields compareMatchingFields;
struct CompareFieldPresence1 compareFieldPresence1;
// expected-error@first.h:* {{'CompareFieldPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field}}
// expected-note@second.h:* {{but in 'Second' found end of class}}
struct CompareFieldPresence2 compareFieldPresence2;
// expected-error@second.h:* {{'CompareFieldPresence2::fieldPresence2' from module 'Second' is not present in definition of 'struct CompareFieldPresence2' in module 'First.Hidden'}}
// expected-note@first.h:* {{definition has no member 'fieldPresence2'}}
struct CompareFieldName compareFieldName;
// expected-error@second.h:* {{'CompareFieldName::fieldNameB' from module 'Second' is not present in definition of 'struct CompareFieldName' in module 'First.Hidden'}}
// expected-note@first.h:* {{definition has no member 'fieldNameB'}}
struct CompareFieldOrder compareFieldOrder;
// expected-error@first.h:* {{'CompareFieldOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'fieldOrderX'}}
// expected-note@second.h:* {{but in 'Second' found field 'fieldOrderY'}}
#endif

#if defined(FIRST)
struct CompareFieldType {
  int fieldType;
};

typedef int FieldTypedefNameA;
struct CompareFieldTypedefName {
  FieldTypedefNameA fieldTypedefName;
};

typedef int TypedefUnderlyingType;
struct CompareFieldTypeUnderlyingTypedef {
  TypedefUnderlyingType fieldTypeUnderlyingTypedef;
};

typedef int TypedefFinal;
struct CompareFieldTypedefChain {
  TypedefFinal fieldTypeTypedefChain;
};
#elif defined(SECOND)
struct CompareFieldType {
  float fieldType;
};

typedef int FieldTypedefNameB;
struct CompareFieldTypedefName {
  FieldTypedefNameB fieldTypedefName;
};

struct CompareFieldTypeUnderlyingTypedef {
  int fieldTypeUnderlyingTypedef;
};

typedef int TypedefIntermediate;
typedef TypedefIntermediate TypedefFinal;
struct CompareFieldTypedefChain {
  TypedefFinal fieldTypeTypedefChain;
};
#else
struct CompareFieldType compareFieldType;
// expected-error@second.h:* {{'CompareFieldType::fieldType' from module 'Second' is not present in definition of 'struct CompareFieldType' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'fieldType' does not match}}
struct CompareFieldTypedefName compareFieldTypedefName;
// expected-error@first.h:* {{'CompareFieldTypedefName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'fieldTypedefName' with type 'FieldTypedefNameA' (aka 'int')}}
// expected-note@second.h:* {{but in 'Second' found field 'fieldTypedefName' with type 'FieldTypedefNameB' (aka 'int')}}
struct CompareFieldTypeUnderlyingTypedef compareFieldTypeUnderlyingTypedef;
// expected-error@first.h:* {{'CompareFieldTypeUnderlyingTypedef' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'fieldTypeUnderlyingTypedef' with type 'TypedefUnderlyingType' (aka 'int')}}
// expected-note@second.h:* {{but in 'Second' found field 'fieldTypeUnderlyingTypedef' with type 'int'}}
struct CompareFieldTypedefChain compareFieldTypedefChain;
#endif

#if defined(FIRST)
struct CompareMatchingBitfields {
  unsigned matchingBitfields : 3;
};

struct CompareBitfieldPresence1 {
  unsigned bitfieldPresence1 : 1;
};
struct CompareBitfieldPresence2 {
  unsigned bitfieldPresence2;
};

struct CompareBitfieldWidth {
  unsigned bitfieldWidth : 2;
};

struct CompareBitfieldWidthExpression {
  unsigned bitfieldWidthExpression : 1 + 1;
};
#elif defined(SECOND)
struct CompareMatchingBitfields {
  unsigned matchingBitfields : 3;
};

struct CompareBitfieldPresence1 {
  unsigned bitfieldPresence1;
};
struct CompareBitfieldPresence2 {
  unsigned bitfieldPresence2 : 1;
};

struct CompareBitfieldWidth {
  unsigned bitfieldWidth : 1;
};

struct CompareBitfieldWidthExpression {
  unsigned bitfieldWidthExpression : 2;
};
#else
struct CompareMatchingBitfields compareMatchingBitfields;
struct CompareBitfieldPresence1 compareBitfieldPresence1;
// expected-error@first.h:* {{'CompareBitfieldPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found bit-field 'bitfieldPresence1'}}
// expected-note@second.h:* {{but in 'Second' found non-bit-field 'bitfieldPresence1'}}
struct CompareBitfieldPresence2 compareBitfieldPresence2;
// expected-error@first.h:* {{'CompareBitfieldPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found non-bit-field 'bitfieldPresence2'}}
// expected-note@second.h:* {{but in 'Second' found bit-field 'bitfieldPresence2'}}
struct CompareBitfieldWidth compareBitfieldWidth;
// expected-error@first.h:* {{'CompareBitfieldWidth' has different definitions in different modules; first difference is definition in module 'First.Hidden' found bit-field 'bitfieldWidth' with one width expression}}
// expected-note@second.h:* {{but in 'Second' found bit-field 'bitfieldWidth' with different width expression}}
struct CompareBitfieldWidthExpression compareBitfieldWidthExpression;
// expected-error@first.h:* {{'CompareBitfieldWidthExpression' has different definitions in different modules; first difference is definition in module 'First.Hidden' found bit-field 'bitfieldWidthExpression' with one width expression}}
// expected-note@second.h:* {{but in 'Second' found bit-field 'bitfieldWidthExpression' with different width expressio}}
#endif

#if defined(FIRST)
struct CompareMatchingArrayFields {
  int matchingArrayField[7];
};

struct CompareArrayLength {
  int arrayLengthField[5];
};

struct CompareArrayType {
  int arrayTypeField[5];
};
#elif defined(SECOND)
struct CompareMatchingArrayFields {
  int matchingArrayField[7];
};

struct CompareArrayLength {
  int arrayLengthField[7];
};

struct CompareArrayType {
  float arrayTypeField[5];
};
#else
struct CompareMatchingArrayFields compareMatchingArrayFields;
struct CompareArrayLength compareArrayLength;
// expected-error@second.h:* {{'CompareArrayLength::arrayLengthField' from module 'Second' is not present in definition of 'struct CompareArrayLength' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'arrayLengthField' does not match}}
struct CompareArrayType compareArrayType;
// expected-error@second.h:* {{'CompareArrayType::arrayTypeField' from module 'Second' is not present in definition of 'struct CompareArrayType' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'arrayTypeField' does not match}}
#endif

#if defined(FIRST)
struct CompareFieldAsForwardDeclaration {
  struct FieldForwardDeclaration *fieldForwardDeclaration;
};

enum FieldEnumA { kFieldEnumValue };
struct CompareFieldAsEnum {
  enum FieldEnumA fieldEnum;
};

struct FieldStructA {};
struct CompareFieldAsStruct {
  struct FieldStructA fieldStruct;
};
#elif defined(SECOND)
struct FieldForwardDeclaration {};
struct CompareFieldAsForwardDeclaration {
  struct FieldForwardDeclaration *fieldForwardDeclaration;
};

enum FieldEnumB { kFieldEnumValue };
struct CompareFieldAsEnum {
  enum FieldEnumB fieldEnum;
};

struct FieldStructB {};
struct CompareFieldAsStruct {
  struct FieldStructB fieldStruct;
};
#else
struct CompareFieldAsForwardDeclaration compareFieldAsForwardDeclaration;
struct CompareFieldAsEnum compareFieldAsEnum;
// expected-error@second.h:* {{'CompareFieldAsEnum::fieldEnum' from module 'Second' is not present in definition of 'struct CompareFieldAsEnum' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'fieldEnum' does not match}}
struct CompareFieldAsStruct compareFieldAsStruct;
// expected-error@second.h:* {{'CompareFieldAsStruct::fieldStruct' from module 'Second' is not present in definition of 'struct CompareFieldAsStruct' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'fieldStruct' does not match}}
#endif

#if defined(FIRST)
union CompareMatchingUnionFields {
  int matchingFieldA;
  float matchingFieldB;
};

union CompareUnionFieldOrder {
  int unionFieldOrderA;
  float unionFieldOrderB;
};

union CompareUnionFieldType {
  int unionFieldType;
};
#elif defined(SECOND)
union CompareMatchingUnionFields {
  int matchingFieldA;
  float matchingFieldB;
};

union CompareUnionFieldOrder {
  float unionFieldOrderB;
  int unionFieldOrderA;
};

union CompareUnionFieldType {
  unsigned int unionFieldType;
};
#else
union CompareMatchingUnionFields compareMatchingUnionFields;
union CompareUnionFieldOrder compareUnionFieldOrder;
// expected-error@first.h:* {{'CompareUnionFieldOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'unionFieldOrderA'}}
// expected-note@second.h:* {{but in 'Second' found field 'unionFieldOrderB'}}
union CompareUnionFieldType compareUnionFieldType;
// expected-error@second.h:* {{'CompareUnionFieldType::unionFieldType' from module 'Second' is not present in definition of 'union CompareUnionFieldType' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'unionFieldType' does not match}}
#endif

// Test that we find and compare definitions even if they are not the first encountered declaration in a module.
#if defined(FIRST)
struct CompareDefinitionsRegardlessForwardDeclarations {
  int definitionField;
};
#elif defined(SECOND)
struct CompareDefinitionsRegardlessForwardDeclarations;
struct CompareDefinitionsRegardlessForwardDeclarations {
  float definitionField;
};
#else
struct CompareDefinitionsRegardlessForwardDeclarations compareDefinitions;
// expected-error@second.h:* {{'CompareDefinitionsRegardlessForwardDeclarations::definitionField' from module 'Second' is not present in definition of 'struct CompareDefinitionsRegardlessForwardDeclarations' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'definitionField' does not match}}
#endif

//--- include/first-nested-struct.h
struct CompareNestedStruct {
  struct NestedLevel1 {
    struct NestedLevel2 {
      int a;
    } y;
  } x;
};

struct IndirectStruct {
  int mismatchingField;
};
struct DirectStruct {
  struct IndirectStruct indirectField;
};
struct CompareDifferentFieldInIndirectStruct {
  struct DirectStruct directField;
};
struct CompareIndirectStructPointer {
  struct DirectStruct *directFieldPointer;
};

//--- include/second-nested-struct.h
struct CompareNestedStruct {
  struct NestedLevel1 {
    struct NestedLevel2 {
      float b;
    } y;
  } x;
};

struct IndirectStruct {
  float mismatchingField;
};
struct DirectStruct {
  struct IndirectStruct indirectField;
};
struct CompareDifferentFieldInIndirectStruct {
  struct DirectStruct directField;
};
struct CompareIndirectStructPointer {
  struct DirectStruct *directFieldPointer;
};

//--- test-nested-struct.c
#include "first-empty.h"
#include "second-nested-struct.h"

#if defined(CASE1)
struct CompareNestedStruct compareNestedStruct;
// expected-error@second-nested-struct.h:* {{'NestedLevel2::b' from module 'Second' is not present in definition of 'struct NestedLevel2' in module 'First.Hidden'}}
// expected-note@first-nested-struct.h:* {{definition has no member 'b'}}
#elif defined(CASE2)
struct CompareDifferentFieldInIndirectStruct compareIndirectStruct;
// expected-error@second-nested-struct.h:* {{'IndirectStruct::mismatchingField' from module 'Second' is not present in definition of 'struct IndirectStruct' in module 'First.Hidden'}}
// expected-note@first-nested-struct.h:* {{declaration of 'mismatchingField' does not match}}
#elif defined(CASE3)
// expected-error@second-nested-struct.h:* {{'IndirectStruct::mismatchingField' from module 'Second' is not present in definition of 'struct IndirectStruct' in module 'First.Hidden'}}
// expected-note@first-nested-struct.h:* {{declaration of 'mismatchingField' does not match}}
struct CompareIndirectStructPointer compareIndirectStructPointer;
struct DirectStruct test() {
  // Make sure the type behind the pointer is inspected.
  return *compareIndirectStructPointer.directFieldPointer;
}
#endif

//--- include/first-anonymous.h
struct CompareAnonymousNestedUnion {
  union {
    int anonymousNestedUnionField;
  };
};

struct CompareAnonymousNestedStruct {
  struct {
    int anonymousNestedStructField;
  };
};

struct CompareDeeplyNestedAnonymousUnionsAndStructs {
  union {
    int x;
    union {
      int y;
      struct {
        int z;
      };
    };
  };
};

//--- include/second-anonymous.h
struct CompareAnonymousNestedUnion {
  union {
    float anonymousNestedUnionField;
  };
};

struct CompareAnonymousNestedStruct {
  struct {
    float anonymousNestedStructField;
  };
};

struct CompareDeeplyNestedAnonymousUnionsAndStructs {
  union {
    int x;
    union {
      int y;
      struct {
        float z;
      };
    };
  };
};

//--- test-anonymous.c
#include "first-empty.h"
#include "second-anonymous.h"

#if defined(CASE1)
struct CompareAnonymousNestedUnion compareAnonymousNestedUnion;
// expected-error-re@second-anonymous.h:* {{'CompareAnonymousNestedUnion::(anonymous union)::anonymousNestedUnionField' from module 'Second' is not present in definition of 'union CompareAnonymousNestedUnion::(anonymous at {{.*}})' in module 'First.Hidden'}}
// expected-note@first-anonymous.h:* {{declaration of 'anonymousNestedUnionField' does not match}}
#elif defined(CASE2)
struct CompareAnonymousNestedStruct compareAnonymousNestedStruct;
// expected-error-re@second-anonymous.h:* {{'CompareAnonymousNestedStruct::(anonymous struct)::anonymousNestedStructField' from module 'Second' is not present in definition of 'struct CompareAnonymousNestedStruct::(anonymous at {{.*}})' in module 'First.Hidden'}}
// expected-note@first-anonymous.h:* {{declaration of 'anonymousNestedStructField' does not match}}
#elif defined(CASE3)
struct CompareDeeplyNestedAnonymousUnionsAndStructs compareDeeplyNested;
// expected-error-re@second-anonymous.h:* {{'CompareDeeplyNestedAnonymousUnionsAndStructs::(anonymous union)::(anonymous union)::(anonymous struct)::z' from module 'Second' is not present in definition of 'struct CompareDeeplyNestedAnonymousUnionsAndStructs::(anonymous at {{.*}})' in module 'First.Hidden'}}
// expected-note@first-anonymous.h:* {{declaration of 'z' does not match}}
#endif
