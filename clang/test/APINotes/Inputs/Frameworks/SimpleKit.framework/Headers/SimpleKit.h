struct RenamedAgainInAPINotesA {
  int field;
} __attribute__((swift_name("bad")));

struct __attribute__((swift_name("bad"))) RenamedAgainInAPINotesB {
  int field;
};
