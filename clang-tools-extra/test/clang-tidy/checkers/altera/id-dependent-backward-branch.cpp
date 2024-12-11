// RUN: %check_clang_tidy %s altera-id-dependent-backward-branch %t -- -header-filter=.* "--" -cl-std=CLC++1.0 -c

void error() {
  // ==== Conditional Expressions ====
  int accumulator = 0;
  for (int i = 0; i < get_local_id(0); i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  int j = 0;
  while (j < get_local_id(0)) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < get_local_id(0));
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]

  // ==== Assignments ====
  int ThreadID = get_local_id(0);

  while (j < ThreadID) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: assignment of ID-dependent variable ThreadID
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < ThreadID);
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]
  // CHECK-NOTES: :[[@LINE-12]]:3: note: assignment of ID-dependent variable ThreadID

  struct { int IDDepField; } Example;
  Example.IDDepField = get_local_id(0);

  for (int i = 0; i < Example.IDDepField; i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: assignment of ID-dependent field IDDepField
    accumulator++;
  }

  while (j < Example.IDDepField) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-10]]:3: note: assignment of ID-dependent field IDDepField
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < Example.IDDepField);
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
  // CHECK-NOTES: :[[@LINE-18]]:3: note: assignment of ID-dependent field IDDepField

  // ==== Inferred Assignments ====
  int ThreadID2 = ThreadID * 2;

  for (int i = 0; i < ThreadID2; i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to variable reference to 'ThreadID2' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: inferred assignment of ID-dependent value from ID-dependent variable ThreadID
    accumulator++;
  }

  // ==== Unused Inferred Assignments ====
  int UnusedThreadID = Example.IDDepField; // OK: not used in any loops

  struct { int IDDepField; } UnusedStruct;
  UnusedStruct.IDDepField = ThreadID * 2; // OK: not used in any loops
}

void success() {
  int accumulator = 0;

  for (int i = 0; i < 1000; i++) {
    if (i < get_local_id(0)) {
      accumulator++;
    }
  }
}

template<char... STOP>
void gh55408(char const input[], int pos) {
  while (((input[pos] != STOP) && ...));
}

