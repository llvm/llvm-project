// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -fblocks -Wno-strict-prototypes %s

#define bool _Bool
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock);
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock);
}

void bar3(id(*)());
void foo3(id (*objectCreationBlock)(int)) {
    return bar3(objectCreationBlock);
}

void bar4(id(^)());
void foo4(id (^objectCreationBlock)(int)) {
    return bar4(objectCreationBlock);
}

void bar5(id(^)(void)); // expected-note 3{{passing argument to parameter here}}
void foo5(id (^objectCreationBlock)(bool)) {
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(bool)' to parameter of type 'id (^)(void)'}}
#undef bool
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
#define bool int
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
}

void bar6(id(^)(int));
void foo6(id (^objectCreationBlock)()) {
    return bar6(objectCreationBlock);
}

void foo7(id (^x)(int)) {
  if (x) { }
}

@interface itf
@end

void foo8() {
  void *P = ^(itf x) {};  // expected-error {{interface type 'itf' cannot be passed by value; did you forget * in 'itf'}}
  P = ^itf(int x) {};     // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf() {};          // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf{};             // expected-error {{interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
}


int foo9() {
  typedef void (^DVTOperationGroupScheduler)();
  id _suboperationSchedulers;

  for (DVTOperationGroupScheduler scheduler in _suboperationSchedulers) {
            ;
        }

}

@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

void foo10() {
    void(^myBlock)(void) = ^{
    };
    NSLog(@"%@", myBlock);
}


// In C, enum constants have the type of the underlying integer type, not the
// enumeration they are part of. We pretend the constants have enum type if
// all the returns seem to be playing along.
enum CStyleEnum {
  CSE_Value = 1,
  CSE_Value2 = 2
};
enum CStyleEnum getCSE();
typedef enum CStyleEnum (^cse_block_t)();

void testCStyleEnumInference(bool arg) {
  cse_block_t a;
  enum CStyleEnum value;

  // No warnings here.
  a = ^{ return getCSE(); };
  a = ^{ return value; };

  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    return 1;
  };

  // No warning here.
  a = ^{
    return CSE_Value;
  };

  // No warnings here.
  a = ^{ if (arg) return CSE_Value; else return getCSE();  };
  a = ^{ if (arg) return getCSE();  else return CSE_Value; };
  a = ^{ if (arg) return value;     else return CSE_Value; };

  // These two blocks actually return 'int'
  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    if (arg)
      return 1;
    else
      return CSE_Value;
  };

  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    if (arg)
      return CSE_Value;
    else
      return 1;
  };

  a = ^{ // expected-error {{incompatible block pointer types assigning to 'cse_block_t' (aka 'enum CStyleEnum (^)()') from 'int (^)(void)'}}
    if (arg)
      return 1;
    else
      return value; // expected-error {{return type 'enum CStyleEnum' must match previous return type 'int'}}
  };

  extern void check_enum(void);
  a = ^{
    return (arg ? (CSE_Value) : (check_enum(), (!arg ? CSE_Value2 : getCSE())));
  };
  a = ^{
    return (arg ? (CSE_Value) : ({check_enum(); CSE_Value2; }));
  };
}


enum FixedTypeEnum : unsigned {
  FTE_Value = 1U
};
enum FixedTypeEnum getFTE();
typedef enum FixedTypeEnum (^fte_block_t)();

void testFixedTypeEnumInference(bool arg) {
  fte_block_t a;
  
  // No warnings here.
  a = ^{ return getFTE(); };

  // Since we fixed the underlying type of the enum, this is considered a
  // compatible block type.
  a = ^{
    return 1U;
  };
  a = ^{
    return FTE_Value;
  };

  // No warnings here.
  a = ^{ if (arg) return FTE_Value; else return FTE_Value; };
  a = ^{ if (arg) return getFTE();  else return getFTE();  };
  a = ^{ if (arg) return FTE_Value; else return getFTE();  };
  a = ^{ if (arg) return getFTE();  else return FTE_Value; };
  
  // These two blocks actually return 'unsigned'.
  a = ^{
    if (arg)
      return 1U;
    else
      return FTE_Value;
  };
  
  a = ^{
    if (arg)
      return FTE_Value;
    else
      return 1U;
  };
}


enum {
  AnonymousValue = 1
};

enum : short {
  FixedAnonymousValue = 1
};

typedef enum {
  TDE_Value
} TypeDefEnum;
TypeDefEnum getTDE();

typedef enum : short {
  TDFTE_Value
} TypeDefFixedTypeEnum;
TypeDefFixedTypeEnum getTDFTE();

typedef int (^int_block_t)();
typedef short (^short_block_t)();
void testAnonymousEnumTypes(int arg) {
  int_block_t IB;
  IB = ^{ return AnonymousValue; };
  IB = ^{ if (arg) return TDE_Value; else return getTDE(); };
  IB = ^{ if (arg) return getTDE(); else return TDE_Value; };

  // Since we fixed the underlying type of the enum, these are considered
  // compatible block types anyway.
  short_block_t SB;
  SB = ^{ return FixedAnonymousValue; };
  SB = ^{ if (arg) return TDFTE_Value; else return getTDFTE(); };
  SB = ^{ if (arg) return getTDFTE(); else return TDFTE_Value; };
}

static inline void inlinefunc() {
  ^{}();
}
void inlinefunccaller() { inlinefunc(); }
