struct RenamedAgainInAPINotesA {
  int field;
} __attribute__((swift_name("bad")));

struct __attribute__((swift_name("bad"))) RenamedAgainInAPINotesB {
  int field;
};

void *getCFOwnedToUnowned(void) __attribute__((cf_returns_retained));
void *getCFUnownedToOwned(void) __attribute__((cf_returns_not_retained));
void *getCFOwnedToNone(void) __attribute__((cf_returns_retained));
id getObjCOwnedToUnowned(void) __attribute__((ns_returns_retained));
id getObjCUnownedToOwned(void) __attribute__((ns_returns_not_retained));

int indirectGetCFOwnedToUnowned(void **out __attribute__((cf_returns_retained)));
int indirectGetCFUnownedToOwned(void **out __attribute__((cf_returns_not_retained)));
int indirectGetCFOwnedToNone(void **out __attribute__((cf_returns_retained)));
int indirectGetCFNoneToOwned(void **out);

@interface MethodTest
- (id)getOwnedToUnowned __attribute__((ns_returns_retained));
- (id)getUnownedToOwned __attribute__((ns_returns_not_retained));
@end
