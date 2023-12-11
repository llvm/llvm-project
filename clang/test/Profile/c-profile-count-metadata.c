// Copy from c-unprofiled-blocks.c but testing the `MD_prof_count`, which will
// will generate the MD node no matter the code is dead or not.

// RUN: llvm-profdata merge %S/Inputs/c-profile-count-metadata.proftext -o %t.profdata
// RUN: %clang_cc1 -mllvm -enable-profile-count-metadata -triple x86_64-apple-macosx10.9 \
// RUN: -main-file-name c-profile-count-metadata.c %s -o - \
// RUN: -emit-llvm -fprofile-instrument-use-path=%t.profdata | FileCheck -check-prefix=PGOUSE %s

// PGOUSE-LABEL: @never_called(i32 noundef %i)
int never_called(int i) {
  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}{{$}}
  if (i) {}

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof.count !{{[0-9]+}}{{$}}
  for (i = 0; i < 100; ++i) {
  }

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof.count !{{[0-9]+}}{{$}}
  while (--i) {}

  // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !llvm.loop [[LOOP1:!.*]], !prof.count !{{[0-9]+}}
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

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof.count !{{[0-9]+}}{{$}}
    for (i = 0; i < 100; ++i) {
    }

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !prof.count !{{[0-9]+}}{{$}}
    while (--i) {}

    // PGOUSE: br i1 %{{[^,]*}}, label %{{[^,]*}}, label %{{[^,]*}}, !llvm.loop [[LOOP2:!.*]], !prof.count !{{[0-9]+}}
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

// PGOUSE: !{{[0-9]+}} = !{!"profile_count", i64 0}
