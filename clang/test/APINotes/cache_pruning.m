// We need 'touch' and 'find' for this test to work.
// REQUIRES: shell

// RUN: rm -rf %t/APINotesCache

// Run Clang. This should generated the cached versions of both and a timestamp.
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DINCLUDE_HEADERLIB
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "APINotes.timestamp"

// Set the timestamp back a very long time. We should try to prune,
// but nothing gets pruned because the API Notes files are new enough.
// RUN: touch -m -a -t 201101010000 %t/APINotes.timestamp 
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "APINotes.timestamp"

// Set the HeaderLib access time back a very long time.
// This shouldn't prune anything, because the timestamp has been updated, so
// the pruning mechanism won't fire.
// RUN: find %t/APINotesCache -name APINotes-*.apinotesc | xargs touch -a -t 201101010000
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "APINotes.timestamp"

// Set the timestack back a very long time. This should prune the
// HeaderLib file, because the pruning mechanism should fire and
// HeaderLib is both old and not used.
// RUN: touch -m -a -t 201101010000 %t/APINotesCache/APINotes.timestamp 
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: ls %t/APINotesCache | not grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "APINotes.timestamp"

// Run Clang. This should generated the cached versions of both and a timestamp.
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DINCLUDE_HEADERLIB
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "APINotes.timestamp"

#ifdef INCLUDE_HEADERLIB
#include "HeaderLib.h"
#endif
#include <SomeKit/SomeKit.h>

int main() { return 0; }
