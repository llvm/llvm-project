// RUN: rm -f %t.log
// RUN: env RC_DEBUG_OPTIONS=1 \
// RUN:     CC_LOG_DIAGNOSTICS=1 CC_LOG_DIAGNOSTICS_FILE=%t.log \
// RUN: %clang -Wfoobar --target=x86_64-apple-darwin11 -fsyntax-only %s
// RUN: FileCheck %s < %t.log

int;

// CHECK: <dict>
// CHECK:   <key>main-file</key>
// CHECK:   <string>{{.*}}cc-log-diagnostics.c</string>
// CHECK:   <key>dwarf-debug-flags</key>
// CHECK:   <string>{{.*}}-Wfoobar{{.*}}-fsyntax-only{{.*}}</string>
// CHECK:   <key>diagnostics</key>
// CHECK:   <array>
// CHECK:     <dict>
// CHECK:       <key>level</key>
// CHECK:       <string>warning</string>
// CHECK:       <key>message</key>
// CHECK:       <string>unknown warning option &apos;-Wfoobar&apos;; did you mean &apos;-W{{.*}}&apos;?</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:       <key>level</key>
// CHECK:       <string>warning</string>
// CHECK:       <key>filename</key>
// CHECK:       <string>{{.*}}cc-log-diagnostics.c</string>
// CHECK:       <key>line</key>
// CHECK:       <integer>7</integer>
// CHECK:       <key>column</key>
// CHECK:       <integer>1</integer>
// CHECK:       <key>message</key>
// CHECK:       <string>declaration does not declare anything</string>
// CHECK:     </dict>
// CHECK:   </array>
// CHECK: </dict>
