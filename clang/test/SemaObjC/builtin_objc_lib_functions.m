// RUN: %clang_cc1 -x objective-c %s -fsyntax-only -verify
Class f0(void) { return objc_getClass("a"); } // expected-error {{call to undeclared library function 'objc_getClass', will assume it exists with type 'id (const char *)'; ISO C99 and later do not support implicit function declarations}} \
					  // expected-note {{include the header <objc/runtime.h> or explicitly provide a declaration for 'objc_getClass'}}

Class f1(void) { return objc_getMetaClass("a"); } // expected-error {{call to undeclared library function 'objc_getMetaClass', will assume it exists with type 'id (const char *)'; ISO C99 and later do not support implicit function declarations}} \
					  // expected-note {{include the header <objc/runtime.h> or explicitly provide a declaration for 'objc_getMetaClass'}}

void f2(id val) { objc_enumerationMutation(val); } // expected-error {{call to undeclared library function 'objc_enumerationMutation', will assume it exists with type 'void (id)'; ISO C99 and later do not support implicit function declarations}} \
						   // expected-note {{include the header <objc/runtime.h> or explicitly provide a declaration for 'objc_enumerationMutation'}}

long double f3(id self, SEL op) { return objc_msgSend_fpret(self, op); } // expected-error {{call to undeclared library function 'objc_msgSend_fpret', will assume it exists with type 'long double (id, SEL, ...)'; ISO C99 and later do not support implicit function declarations}} \
    // expected-note {{include the header <objc/message.h> or explicitly provide a declaration for 'objc_msgSend_fpret'}}

id f4(struct objc_super *super, SEL op) { // expected-warning {{declaration of 'struct objc_super' will not be visible outside of this function}}
  return objc_msgSendSuper(super, op); // expected-error {{call to undeclared library function 'objc_msgSendSuper', will assume it exists with type 'id (struct objc_super *, SEL, ...)'; ISO C99 and later do not support implicit function declarations}} \
					// expected-note {{include the header <objc/message.h> or explicitly provide a declaration for 'objc_msgSendSuper'}}
}

id f5(id val, id *dest) {
  return objc_assign_strongCast(val, dest); // expected-error {{call to undeclared library function 'objc_assign_strongCast', will assume it exists with type 'id (id, id *)'; ISO C99 and later do not support implicit function declarations}} \
					    // expected-note {{include the header <objc/objc-auto.h> or explicitly provide a declaration for 'objc_assign_strongCast'}}
}

int f6(Class exceptionClass, id exception) {
  return objc_exception_match(exceptionClass, exception); // expected-error {{call to undeclared library function 'objc_exception_match', will assume it exists with type 'int (id, id)'; ISO C99 and later do not support implicit function declarations}} \
  							  // expected-note {{include the header <objc/objc-exception.h> or explicitly provide a declaration for 'objc_exception_match'}}
}
