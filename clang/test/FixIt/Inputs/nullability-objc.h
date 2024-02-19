@class Item;
@class Container<ObjectType>;
@protocol Protocol;

// The first pointer in the file is handled in a different way so need
// a separate test for this case even if the parameter type is the same as in
// objcIdParameterWithProtocol.
void objcIdParameterWithProtocolFirstInFile(id<Protocol> i); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:57-[[@LINE-3]]:57}:" _Nullable"
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:57-[[@LINE-4]]:57}:" _Nonnull"

int * _Nonnull forceNullabilityWarningsObjC(void);

void objcClassParameter(Item *i); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:31-[[@LINE-3]]:31}:" _Nullable "
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:31-[[@LINE-4]]:31}:" _Nonnull "

void objcClassParameterWithProtocol(Item<Protocol> *i); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:53-[[@LINE-3]]:53}:" _Nullable "
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:53-[[@LINE-4]]:53}:" _Nonnull "

void objcIdParameterWithProtocol(id<Protocol> i); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:46-[[@LINE-3]]:46}:" _Nullable"
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:46-[[@LINE-4]]:46}:" _Nonnull"

// Class parameters don't have nullability type specifier.
void objcParameterizedClassParameter(Container<Item *> *c); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:57-[[@LINE-3]]:57}:" _Nullable "
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:57-[[@LINE-4]]:57}:" _Nonnull "

// Class parameters don't have nullability type specifier.
void objcParameterizedClassParameterWithProtocol(Container<id<Protocol>> *c); // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-3]]:75-[[@LINE-3]]:75}:" _Nullable "
// CHECK: fix-it:"{{.*}}nullability-objc.h":{[[@LINE-4]]:75-[[@LINE-4]]:75}:" _Nonnull "
