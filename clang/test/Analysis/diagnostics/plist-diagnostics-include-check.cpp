// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -analyzer-output=plist-multi-file %s -o %t.plist
// RUN: FileCheck --input-file=%t.plist %s

#include "Inputs/include/plist-diagnostics-include-check-macro.h"

void foo() {
  PlistCheckMacro()
#define PLIST_DEF_MACRO .run();
#include "Inputs/include/plist-diagnostics-include-check-macro.def"
}

// CHECK:   <key>diagnostics</key>
// CHECK-NEXT:  <array>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>7</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>7</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>1</integer>
// CHECK-NEXT:          <key>col</key><integer>15</integer>
// CHECK-NEXT:          <key>file</key><integer>2</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Calling &apos;PlistCheckMacro::run&apos;</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Calling &apos;PlistCheckMacro::run&apos;</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>6</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>1</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>depth</key><integer>1</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Entered call from &apos;foo&apos;</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Entered call from &apos;foo&apos;</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>6</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>1</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>6</integer>
// CHECK-NEXT:            <key>col</key><integer>6</integer>
// CHECK-NEXT:            <key>file</key><integer>1</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>7</integer>
// CHECK-NEXT:            <key>col</key><integer>5</integer>
// CHECK-NEXT:            <key>file</key><integer>1</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>7</integer>
// CHECK-NEXT:            <key>col</key><integer>32</integer>
// CHECK-NEXT:            <key>file</key><integer>1</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>7</integer>
// CHECK-NEXT:       <key>col</key><integer>5</integer>
// CHECK-NEXT:       <key>file</key><integer>1</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>7</integer>
// CHECK-NEXT:          <key>col</key><integer>5</integer>
// CHECK-NEXT:          <key>file</key><integer>1</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>7</integer>
// CHECK-NEXT:          <key>col</key><integer>34</integer>
// CHECK-NEXT:          <key>file</key><integer>1</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>1</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>REACHABLE</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>REACHABLE</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>REACHABLE</string>
// CHECK-NEXT:    <key>category</key><string>debug</string>
// CHECK-NEXT:    <key>type</key><string>Checking analyzer assumptions</string>
// CHECK-NEXT:    <key>check_name</key><string>debug.ExprInspection</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>93b4eab05b21c892c8e31723e5af3f59</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>C++ method</string>
// CHECK-NEXT:   <key>issue_context</key><string>run</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>7</integer>
// CHECK-NEXT:    <key>col</key><integer>5</integer>
// CHECK-NEXT:    <key>file</key><integer>1</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  </array>
