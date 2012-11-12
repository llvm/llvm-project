// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-output=plist -o %t %s
// RUN: FileCheck --input-file %t %s

#include "undef-value-callee.h"

// This code used to cause a crash since we were not adding fileID of the header to the plist diagnostic.

int test_calling_unimportant_callee(int argc, char *argv[]) {
  int x;
  callee();
  return x; // expected-warning {{Undefined or garbage value returned to caller}}
}

//CHECK: <dict>
//CHECK:  <key>files</key>
//CHECK:  <array>
//CHECK:  </array>
//CHECK:  <key>diagnostics</key>
//CHECK:  <array>
//CHECK:   <dict>
//CHECK:    <key>path</key>
//CHECK:    <array>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>event</string>
//CHECK:      <key>location</key>
//CHECK:      <dict>
//CHECK:       <key>line</key><integer>9</integer>
//CHECK:       <key>col</key><integer>3</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>9</integer>
//CHECK:          <key>col</key><integer>3</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>9</integer>
//CHECK:          <key>col</key><integer>7</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>0</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Variable &apos;x&apos; declared without an initial value</string>
//CHECK:      <key>message</key>
//CHECK: <string>Variable &apos;x&apos; declared without an initial value</string>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>control</string>
//CHECK:      <key>edges</key>
//CHECK:       <array>
//CHECK:        <dict>
//CHECK:         <key>start</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>9</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>9</integer>
//CHECK:            <key>col</key><integer>5</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>10</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>10</integer>
//CHECK:            <key>col</key><integer>8</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:        </dict>
//CHECK:       </array>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>control</string>
//CHECK:      <key>edges</key>
//CHECK:       <array>
//CHECK:        <dict>
//CHECK:         <key>start</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>10</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>10</integer>
//CHECK:            <key>col</key><integer>8</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>11</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>11</integer>
//CHECK:            <key>col</key><integer>8</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:        </dict>
//CHECK:       </array>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>event</string>
//CHECK:      <key>location</key>
//CHECK:      <dict>
//CHECK:       <key>line</key><integer>11</integer>
//CHECK:       <key>col</key><integer>3</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>11</integer>
//CHECK:          <key>col</key><integer>10</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>11</integer>
//CHECK:          <key>col</key><integer>10</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>0</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Undefined or garbage value returned to caller</string>
//CHECK:      <key>message</key>
//CHECK: <string>Undefined or garbage value returned to caller</string>
//CHECK:     </dict>
//CHECK:    </array>
//CHECK:    <key>description</key><string>Undefined or garbage value returned to caller</string>
//CHECK:    <key>category</key><string>Logic error</string>
//CHECK:    <key>type</key><string>Garbage return value</string>
//CHECK:   <key>issue_context_kind</key><string>function</string>
//CHECK:   <key>issue_context</key><string>test_calling_unimportant_callee</string>
//CHECK:   <key>issue_hash</key><integer>3</integer>
//CHECK:   <key>location</key>
//CHECK:   <dict>
//CHECK:    <key>line</key><integer>11</integer>
//CHECK:    <key>col</key><integer>3</integer>
//CHECK:    <key>file</key><integer>0</integer>
//CHECK:   </dict>
//CHECK:   </dict>
//CHECK:  </array>
//CHECK: </dict>
//CHECK: </plist>
