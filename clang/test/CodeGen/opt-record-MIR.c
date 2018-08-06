// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-apple-ios -S -o /dev/null %s -O2 -dwarf-column-info -Rpass-missed=regalloc 2>&1 | FileCheck -check-prefix=REMARK %s
// RUN: %clang_cc1 -triple arm64-apple-ios -S -o /dev/null %s -O2 -dwarf-column-info 2>&1 | FileCheck -allow-empty -check-prefix=NO_REMARK %s
// RUN: %clang_cc1 -triple arm64-apple-ios -S -o /dev/null %s -O2 -dwarf-column-info -opt-record-file %t.yaml
// RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

void bar(float);

void foo(float *p, int i) {
  while (i--)  {
    float f = *p;
    asm("" ::
        : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "fp", "lr", "sp", "memory");
    bar(f);
  }
}

// REMARK: opt-record-MIR.c:10:11: remark: {{.}} spills {{.}} reloads generated in loop
// NO_REMARK-NOT: remark:

// YAML: --- !Missed
// YAML: Pass:            regalloc
// YAML: Name:            LoopSpillReload
// YAML: DebugLoc:        { File: {{[^,]+}},
// YAML:                    Line: 10,
// YAML:                    Column: 11 }
// YAML: Function:        foo
// YAML: Args:
// YAML:   - NumSpills:       '{{.}}'
// YAML:   - String:          ' spills '
// YAML:   - NumReloads:      '{{.}}'
// YAML:   - String:          ' reloads '
// YAML:   - String:          generated
// YAML: ...
