// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.VirtualCall \
// RUN:     -analyzer-config optin.cplusplus.VirtualCall:ShowFixIts=true \
// RUN:     %s 2>&1 | FileCheck -check-prefix=TEXT %s

// RUN: %check_analyzer_fixit %s %t \
// RUN:     -analyzer-checker=core,optin.cplusplus.VirtualCall \
// RUN:     -analyzer-config optin.cplusplus.VirtualCall:ShowFixIts=true \
// RUN:     -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck -check-prefix=PLIST %s

struct S {
  virtual void foo();
  S() {
    foo();
    // expected-warning@-1{{Call to virtual method 'S::foo' during construction bypasses virtual dispatch}}
    // CHECK-FIXES:      S() {
    // CHECK-FIXES-NEXT:   S::foo();
    // CHECK-FIXES-NEXT: }
  }
  ~S();
};

// TEXT: warning: Call to virtual method 'S::foo' during construction
// TEXT-SAME: bypasses virtual dispatch
// TEXT-NEXT: foo();
// TEXT-NEXT: ^~~~~
// TEXT-NEXT: S::
// TEXT-NEXT: 1 warning generated.

// PLIST:  <key>fixits</key>
// PLIST-NEXT:  <array>
// PLIST-NEXT:   <dict>
// PLIST-NEXT:    <key>remove_range</key>
// PLIST-NEXT:    <array>
// PLIST-NEXT:     <dict>
// PLIST-NEXT:      <key>line</key><integer>14</integer>
// PLIST-NEXT:      <key>col</key><integer>5</integer>
// PLIST-NEXT:      <key>file</key><integer>0</integer>
// PLIST-NEXT:     </dict>
// PLIST-NEXT:     <dict>
// PLIST-NEXT:      <key>line</key><integer>14</integer>
// PLIST-NEXT:      <key>col</key><integer>4</integer>
// PLIST-NEXT:      <key>file</key><integer>0</integer>
// PLIST-NEXT:     </dict>
// PLIST-NEXT:    </array>
// PLIST-NEXT:    <key>insert_string</key><string>S::</string>
// PLIST-NEXT:   </dict>
// PLIST-NEXT:  </array>
