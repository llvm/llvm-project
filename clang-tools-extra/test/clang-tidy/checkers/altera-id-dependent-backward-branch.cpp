// RUN: %check_clang_tidy %s altera-id-dependent-backward-branch %t -- -header-filter=.* "--" -cl-std=CL1.2 -c --include opencl-c.h

typedef struct ExampleStruct {
  int IDDepField;
} ExampleStruct;

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
    // CHECK-NOTES: :[[@LINE-3]]:3: note: assignment of ID-dependent variable ThreadID
    // CHECK-NOTES: :[[@LINE-2]]:10: warning: backward branch (while loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  ExampleStruct Example;
  Example.IDDepField = get_local_id(0);

  // ==== Inferred Assignments ====
  int ThreadID2 = ThreadID * get_local_size(0);

  int ThreadID3 = Example.IDDepField; // OK: not used in any loops

  ExampleStruct UnusedStruct = {
      ThreadID * 2 // OK: not used in any loops
  };

  for (int i = 0; i < ThreadID2; i++) {
    // CHECK-NOTES: :[[@LINE-9]]:3: note: inferred assignment of ID-dependent value from ID-dependent variable ThreadID
    // CHECK-NOTES: :[[@LINE-2]]:19: warning: backward branch (for loop) is ID-dependent due to variable reference to 'ThreadID2' and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < ThreadID);
  // CHECK-NOTES: :[[@LINE-29]]:3: note: assignment of ID-dependent variable ThreadID
  // CHECK-NOTES: :[[@LINE-2]]:12: warning: backward branch (do loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]

  for (int i = 0; i < Example.IDDepField; i++) {
    // CHECK-NOTES: :[[@LINE-24]]:3: note: assignment of ID-dependent field IDDepField
    // CHECK-NOTES: :[[@LINE-2]]:19: warning: backward branch (for loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  while (j < Example.IDDepField) {
    // CHECK-NOTES: :[[@LINE-30]]:3: note: assignment of ID-dependent field IDDepField
    // CHECK-NOTES: :[[@LINE-2]]:10: warning: backward branch (while loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < Example.IDDepField);
  // CHECK-NOTES: :[[@LINE-38]]:3: note: assignment of ID-dependent field IDDepField
  // CHECK-NOTES: :[[@LINE-2]]:12: warning: backward branch (do loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
}

void success() {
  int accumulator = 0;

  for (int i = 0; i < 1000; i++) {
    if (i < get_local_id(0)) {
      accumulator++;
    }
  }
}
