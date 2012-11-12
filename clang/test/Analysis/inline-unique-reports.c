// RUN: %clang --analyze %s -o %t > /dev/null 2>&1
// RUN: FileCheck -input-file %t %s

static inline bug(int *p) {
  *p = 0xDEADBEEF;
}

void test_bug_1() {
  int *p = 0;
  bug(p);
}

void test_bug_2() {
  int *p = 0;
  bug(p);
}

// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <plist version="1.0">
// CHECK: <dict>
// CHECK:  <key>files</key>
// CHECK:  <array>
// CHECK:  </array>
// CHECK:  <key>diagnostics</key>
// CHECK:  <array>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>15</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>15</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>15</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>15</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>15</integer>
// CHECK:          <key>col</key><integer>8</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Calling &apos;bug&apos;</string>
// CHECK:      <key>message</key>
// CHECK: <string>Calling &apos;bug&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>4</integer>
// CHECK:       <key>col</key><integer>1</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>depth</key><integer>1</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Entered call from &apos;test_bug_2&apos;</string>
// CHECK:      <key>message</key>
// CHECK: <string>Entered call from &apos;test_bug_2&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>4</integer>
// CHECK:            <key>col</key><integer>1</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>4</integer>
// CHECK:            <key>col</key><integer>6</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>5</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>5</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>5</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>1</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Dereference of null pointer</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>bug</string>
// CHECK:   <key>issue_hash</key><integer>1</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>5</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>

