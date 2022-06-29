// RUN: %check_cir_tidy %s cir-lifetime-check %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: cir-lifetime-check.RemarksList, value: "None"}, \
// RUN:   {key: cir-lifetime-check.HistoryList, value: "invalid;null"}]}' \
// RUN: --

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }        // CHECK-NOTES: note: pointee 'x' invalidated at end of scope
  *p = 42; // CHECK-MESSAGES: warning: use of invalid pointer 'p'
  return p;
}
