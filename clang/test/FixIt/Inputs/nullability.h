int * _Nonnull forceNullabilityWarnings(void);

void arrayParameter(int x[]); // expected-warning {{array parameter is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-3]]:27-[[@LINE-3]]:27}:"_Nullable"
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-4]]:27-[[@LINE-4]]:27}:"_Nonnull"

void arrayParameterWithSize(int x[5]); // expected-warning {{array parameter is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-3]]:35-[[@LINE-3]]:35}:"_Nullable "
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-4]]:35-[[@LINE-4]]:35}:"_Nonnull "

void arrayParameterWithStar(int x[*]); // expected-warning {{array parameter is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable'}}
// expected-note@-2 {{insert '_Nonnull'}}
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-3]]:35-[[@LINE-3]]:35}:"_Nullable "
// CHECK: fix-it:"{{.*}}nullability.h":{[[@LINE-4]]:35-[[@LINE-4]]:35}:"_Nonnull "


// No fix-its on either the macro definition or instantiation.
// CHECK-NOT: fix-it:"{{.*}}nullability.h":{[[@LINE+2]]
// CHECK-NOT: fix-it:"{{.*}}nullability.h":{[[@LINE+2]]
#define PTR(X) X *
PTR(int) a; // expected-warning{{pointer is missing a nullability type specifier}}
#undef PTR
