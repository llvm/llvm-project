// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core.DivideZero -analyzer-config inline-lambdas=true -analyzer-output plist -verify %s -o %t
// RUN: FileCheck --input-file=%t %s


// Diagnostic inside a lambda

void diagnosticFromLambda() {
  int i = 0;
  [=] {
    int p = 5/i; // expected-warning{{Division by zero}}
    (void)p;
  }();
}

// CHECK: <key>diagnostics</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>8</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>8</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>9</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>9</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>9</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>9</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>12</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>The value 0 is assigned to field &apos;&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>The value 0 is assigned to field &apos;&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>9</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>9</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>12</integer>
// CHECK-NEXT:         <key>col</key><integer>5</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling &apos;operator()&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling &apos;operator()&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>9</integer>
// CHECK-NEXT:      <key>col</key><integer>5</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Entered call from &apos;diagnosticFromLambda&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Entered call from &apos;diagnosticFromLambda&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>9</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>9</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>10</integer>
// CHECK-NEXT:           <key>col</key><integer>14</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>10</integer>
// CHECK-NEXT:           <key>col</key><integer>14</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>10</integer>
// CHECK-NEXT:      <key>col</key><integer>14</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>10</integer>
// CHECK-NEXT:         <key>col</key><integer>13</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>10</integer>
// CHECK-NEXT:         <key>col</key><integer>15</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Division by zero</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Division by zero</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Division by zero</string>
// CHECK-NEXT:   <key>category</key><string>Logic error</string>
// CHECK-NEXT:   <key>type</key><string>Division by zero</string>
// CHECK-NEXT:   <key>check_name</key><string>core.DivideZero</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>bd4eed3234018edced5efc2ed5562a74</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>C++ method</string>
// CHECK-NEXT:  <key>issue_context</key><string>operator()</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>10</integer>
// CHECK-NEXT:   <key>col</key><integer>14</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>
