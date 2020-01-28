// RUN: %clang_cc1 -Wall -fsyntax-only %s -diagnostic-log-file %t.diag
// RUN: FileCheck --input-file=%t.diag %s
// RUN: rm -f %t

void foo() {
  int voodoo;
  voodoo = voodoo + 1;
}

// CHECK: <dict>
// CHECK:   <key>main-file</key>
// CHECK:   <string>{{.*}}logfile-diags-fixits.c</string>
// CHECK:   <key>diagnostics</key>
// CHECK:   <array>
// CHECK:     <dict>
// CHECK:       <key>level</key>
// CHECK:       <string>warning</string>
// CHECK:       <key>filename</key>
// CHECK:       <string>{{.*}}logfile-diags-fixits.c</string>
// CHECK:       <key>line</key>
// CHECK:       <integer>7</integer>
// CHECK:       <key>column</key>
// CHECK:       <integer>12</integer>
// CHECK:       <key>message</key>
// CHECK:       <string>variable &apos;voodoo&apos; is uninitialized when used here</string>
// CHECK:       <key>ID</key>
// CHECK:       <integer>{{.*}}</integer>
// CHECK:       <key>WarningOption</key>
// CHECK:       <string>uninitialized</string>
// CHECK:       <key>source-ranges</key>
// CHECK:       <array>
// CHECK:         <dict>
// CHECK:           <key>start-at</key>
// CHECK:           <dict>
// CHECK:             <key>line</key>
// CHECK:             <integer>7</integer>
// CHECK:             <key>column</key>
// CHECK:             <integer>12</integer>
// CHECK:             <key>offset</key>
// CHECK:             <integer>169</integer>
// CHECK:           </dict>
// CHECK:           <key>end-before</key>
// CHECK:           <dict>
// CHECK:             <key>line</key>
// CHECK:             <integer>7</integer>
// CHECK:             <key>column</key>
// CHECK:             <integer>18</integer>
// CHECK:             <key>offset</key>
// CHECK:             <integer>175</integer>
// CHECK:           </dict>
// CHECK:         </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:       <key>level</key>
// CHECK:       <string>note</string>
// CHECK:       <key>filename</key>
// CHECK:       <string>{{.*}}logfile-diags-fixits.c</string>
// CHECK:       <key>line</key>
// CHECK:       <integer>6</integer>
// CHECK:       <key>column</key>
// CHECK:       <integer>13</integer>
// CHECK:       <key>message</key>
// CHECK:       <string>initialize the variable &apos;voodoo&apos; to silence this warning</string>
// CHECK:       <key>ID</key>
// CHECK:       <integer>{{.*}}</integer>
// CHECK:       <key>fixits</key>
// CHECK:       <array>
// CHECK:         <dict>
// CHECK:           <key>start-at</key>
// CHECK:           <dict>
// CHECK:             <key>line</key>
// CHECK:             <integer>6</integer>
// CHECK:             <key>column</key>
// CHECK:             <integer>13</integer>
// CHECK:             <key>offset</key>
// CHECK:             <integer>156</integer>
// CHECK:           </dict>
// CHECK:           <key>end-before</key>
// CHECK:           <dict>
// CHECK:             <key>line</key>
// CHECK:             <integer>6</integer>
// CHECK:             <key>column</key>
// CHECK:             <integer>13</integer>
// CHECK:             <key>offset</key>
// CHECK:             <integer>156</integer>
// CHECK:           </dict>
// CHECK:           <key>replacement</key>
// CHECK:           <string> = 0</string>
// CHECK:         </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:   </array>
// CHECK: </dict>

