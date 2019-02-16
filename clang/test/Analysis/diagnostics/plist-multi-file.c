// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-html -o %t.plist -verify %s
// RUN: FileCheck --input-file=%t.plist %s

#include "plist-multi-file.h"

void bar() {
  foo(0);
}

// CHECK: <key>diagnostics</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>7</integer>
// CHECK-NEXT:      <key>col</key><integer>7</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>7</integer>
// CHECK-NEXT:         <key>col</key><integer>7</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>7</integer>
// CHECK-NEXT:         <key>col</key><integer>7</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Passing null pointer value via 1st parameter &apos;ptr&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Passing null pointer value via 1st parameter &apos;ptr&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>7</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>7</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>7</integer>
// CHECK-NEXT:         <key>col</key><integer>8</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling &apos;foo&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling &apos;foo&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>1</integer>
// CHECK-NEXT:      <key>col</key><integer>1</integer>
// CHECK-NEXT:      <key>file</key><integer>1</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Entered call from &apos;bar&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Entered call from &apos;bar&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>1</integer>
// CHECK-NEXT:           <key>col</key><integer>1</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>1</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>2</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>1</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>2</integer>
// CHECK-NEXT:      <key>col</key><integer>8</integer>
// CHECK-NEXT:      <key>file</key><integer>1</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>2</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>1</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>2</integer>
// CHECK-NEXT:         <key>col</key><integer>6</integer>
// CHECK-NEXT:         <key>file</key><integer>1</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;ptr&apos;)</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;ptr&apos;)</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Dereference of null pointer (loaded from variable &apos;ptr&apos;)</string>
// CHECK-NEXT:   <key>category</key><string>Logic error</string>
// CHECK-NEXT:   <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:   <key>check_name</key><string>core.NullDereference</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>2058c95994cab381890af28e7bf354bf</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>foo</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>2</integer>
// CHECK-NEXT:   <key>col</key><integer>8</integer>
// CHECK-NEXT:   <key>file</key><integer>1</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <key>HTMLDiagnostics_files</key>
// CHECK-NEXT:  <array>
// CHECK-NEXT:   <string>report-{{([0-9a-f]{6})}}.html</string>
// CHECK-NEXT:  </array>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>
