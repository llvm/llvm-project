struct RenamedAgainInAPINotesA {
  int field;
} __attribute__((__swift_name__("bad")));

struct __attribute__((__swift_name__("bad"))) RenamedAgainInAPINotesB {
  int field;
};

void *getCFOwnedToUnowned(void) __attribute__((__cf_returns_retained__));
void *getCFUnownedToOwned(void) __attribute__((__cf_returns_not_retained__));
void *getCFOwnedToNone(void) __attribute__((__cf_returns_retained__));
id getObjCOwnedToUnowned(void) __attribute__((__ns_returns_retained__));
id getObjCUnownedToOwned(void) __attribute__((__ns_returns_not_retained__));

int indirectGetCFOwnedToUnowned(void **out __attribute__((__cf_returns_retained__)));
int indirectGetCFUnownedToOwned(void **out __attribute__((__cf_returns_not_retained__)));
int indirectGetCFOwnedToNone(void **out __attribute__((__cf_returns_retained__)));
int indirectGetCFNoneToOwned(void **out);

#pragma clang arc_cf_code_audited begin
void *getCFAuditedToUnowned_DUMP(void);
void *getCFAuditedToOwned_DUMP(void);
void *getCFAuditedToNone_DUMP(void);
#pragma clang arc_cf_code_audited end

@interface MethodTest
- (id)getOwnedToUnowned __attribute__((__ns_returns_retained__));
- (id)getUnownedToOwned __attribute__((__ns_returns_not_retained__));
@end
