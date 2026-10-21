// RUN: %clang_cc1 -Oz -triple x86_64-unknown-linux-gnu -fexceptions -fsyntax-only %s -verify

typedef __UINT64_TYPE__ uint64_t;

typedef enum _Unwind_Reason_Code _Unwind_Reason_Code;
typedef enum _Unwind_Action _Unwind_Action;

struct _Unwind_Exception;
struct _Unwind_Context;

int __swift_personality_int;
_Unwind_Reason_Code __swift_personality_v0(int version, _Unwind_Action action, uint64_t exception_class, struct _Unwind_Exception *exception, struct _Unwind_Context *context);
_Unwind_Reason_Code __swift_personality_v1(int version, _Unwind_Action action, uint64_t exception_class, struct _Unwind_Exception *exception, struct _Unwind_Context *context);
_Unwind_Reason_Code __swift_personality_v2(int version, _Unwind_Action action, uint64_t exception_class, struct _Unwind_Exception *exception, struct _Unwind_Context *context);

// expected-note@+2{{previous attribute is here}}
// expected-note@+1{{previous attribute is here}}
void __attribute__((__personality__(__swift_personality_v0))) function(void);
// expected-error@+1{{personality routine in declaration does not match previous declaration}}
void __attribute__((__personality__(__swift_personality_v1))) function(void);
// expected-note@+3{{previous attribute is here}}
// expected-error@+2{{personality routine in declaration does not match previous declaration}}
// expected-error@+1{{personality routine in declaration does not match previous declaration}}
void __attribute__((__personality__(__swift_personality_v1))) function(void) __attribute__((__personality__(__swift_personality_v2)));
// expected-error@+1{{'__personality__' argument is not a function}}
void __attribute__((__personality__(__swift_personality_int))) function(void) {
}

// expected-warning@+1{{'__personality__' attribute only applies to functions}}
int variable __attribute__((__personality__(__swift_personality_v0)));
