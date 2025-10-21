// Blocks that we have no profile data for (ie, it was never reached in training
// runs) shouldn't have any branch weight metadata added.

// RUN: llvm-profdata merge %S/Inputs/c-unprofiled-blocks.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-unprofiled-blocks.c %s -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata | FileCheck -check-prefix=PGOUSE %s

// PGOUSE-LABEL: @never_called(i32 noundef %i)
int never_called(int i) {
  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
  if (i) {}

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
  for (i = 0; i < 100; ++i) {
  }

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
  while (--i) {}

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !llvm.loop [[LOOP1:!.*]]
  do {} while (i++ < 75);

  // PGOUSE: switch {{.*}} [
  // PGOUSE-NEXT: i32 12
  // PGOUSE-NEXT: i32 82
  // PGOUSE-NEXT: ]{{$}}
  switch (i) {
  case 12: return 3;
  case 82: return 0;
  default: return 89;
  }
}

// PGOUSE-LABEL: @dead_code(i32 noundef %i)
int dead_code(int i) {
  // PGOUSE: br {{.*}}, !prof !{{[0-9]+}}
  if (i) {
    // This branch is never reached.

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
    if (!i) {}

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
    for (i = 0; i < 100; ++i) {
    }

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
    while (--i) {}

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !llvm.loop [[LOOP2:!.*]]
    do {} while (i++ < 75);

    // PGOUSE: switch {{.*}} [
    // PGOUSE-NEXT: i32 12
    // PGOUSE-NEXT: i32 82
    // PGOUSE-NEXT: ]{{$}}
    switch (i) {
    case 12: return 3;
    case 82: return 0;
    default: return 89;
    }
  }
  return 2;
}

// PGOUSE-LABEL: @main(i32 noundef %argc, ptr noundef %argv)
int main(int argc, const char *argv[]) {
  dead_code(0);
  return 0;
}
