// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CALLABLE_WHEN(...)      __attribute__ ((callable_when(__VA_ARGS__)))
#define CONSUMABLE(state)       __attribute__ ((consumable(state)))
#define SET_TYPESTATE(state)    __attribute__ ((set_typestate(state)))
#define RETURN_TYPESTATE(state) __attribute__ ((return_typestate(state)))
#define TEST_TYPESTATE(state)   __attribute__ ((test_typestate(state)))

// FIXME: This test is here because the warning is issued by the Consumed
//        analysis, not SemaDeclAttr.  The analysis won't run after an error
//        has been issued.  Once the attribute propagation for template
//        instantiation has been fixed, this can be moved somewhere else and the
//        definition can be removed.
int returnTypestateForUnconsumable() RETURN_TYPESTATE(consumed); // expected-warning {{return state set for an unconsumable type 'int'}}
int returnTypestateForUnconsumable() {
  return 0;
}

class AttrTester0 {
  void consumes()       __attribute__ ((set_typestate())); // expected-error {{'set_typestate' attribute takes one argument}}
  bool testUnconsumed() __attribute__ ((test_typestate())); // expected-error {{'test_typestate' attribute takes one argument}}
  void callableWhen()   __attribute__ ((callable_when())); // expected-error {{'callable_when' attribute takes at least 1 argument}}
};

int var0 SET_TYPESTATE(consumed); // expected-warning {{'set_typestate' attribute only applies to functions}}
int var1 TEST_TYPESTATE(consumed); // expected-warning {{'test_typestate' attribute only applies to}}
int var2 CALLABLE_WHEN("consumed"); // expected-warning {{'callable_when' attribute only applies to}}
int var3 CONSUMABLE(consumed); // expected-warning {{'consumable' attribute only applies to classes}}
int var4 RETURN_TYPESTATE(consumed); // expected-warning {{'return_typestate' attribute only applies to functions}}

void function0() SET_TYPESTATE(consumed); // expected-warning {{'set_typestate' attribute only applies to}}
void function1() TEST_TYPESTATE(consumed); // expected-warning {{'test_typestate' attribute only applies to}}
void function2() CALLABLE_WHEN("consumed"); // expected-warning {{'callable_when' attribute only applies to}}
void function3() CONSUMABLE(consumed); // expected-warning {{'consumable' attribute only applies to classes}}

class CONSUMABLE(unknown) AttrTester1 {
  void callableWhen0()  CALLABLE_WHEN("unconsumed");
  void callableWhen1()  CALLABLE_WHEN(42); // expected-error {{'callable_when' attribute requires a string}}
  void callableWhen2()  CALLABLE_WHEN("foo"); // expected-warning {{'callable_when' attribute argument not supported: foo}}
  void callableWhen3()  CALLABLE_WHEN(unconsumed);
  void consumes()       SET_TYPESTATE(consumed);
  bool testUnconsumed() TEST_TYPESTATE(consumed);
};

AttrTester1 returnTypestateTester0() RETURN_TYPESTATE(not_a_state); // expected-warning {{'return_typestate' attribute argument not supported: 'not_a_state'}}
AttrTester1 returnTypestateTester1() RETURN_TYPESTATE(42); // expected-error {{'return_typestate' attribute requires an identifier}}

void returnTypestateTester2(AttrTester1 &Param RETURN_TYPESTATE(unconsumed));

class AttrTester2 {
  void callableWhen()   CALLABLE_WHEN("unconsumed"); // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
  void consumes()       SET_TYPESTATE(consumed); // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
  bool testUnconsumed() TEST_TYPESTATE(consumed); // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
};

class CONSUMABLE(42) AttrTester3; // expected-error {{'consumable' attribute requires an identifier}}


class CONSUMABLE(unconsumed)
      __attribute__((consumable_auto_cast_state))
      __attribute__((consumable_set_state_on_read))
      Status {
};



