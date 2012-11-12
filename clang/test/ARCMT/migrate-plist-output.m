// RUN: %clang_cc1 -arcmt-migrate -mt-migrate-directory %t.dir -arcmt-migrate-report-output %t.plist %s 
// RUN: FileCheck %s -input-file=%t.plist
// RUN: rm -rf %t.dir

@protocol NSObject
- (oneway void)release;
@end

void test(id p) {
  [p release];
}

// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
// CHECK: <plist version="1.0">
// CHECK: <dict>
// CHECK:  <key>files</key>
// CHECK:  <array>
// CHECK:  </array>
// CHECK:  <key>diagnostics</key>
// CHECK:  <array>
// CHECK:   <dict>
// CHECK:    <key>description</key><string>ARC forbids explicit message send of &apos;release&apos;</string>
// CHECK:    <key>category</key><string>ARC Restrictions</string>
// CHECK:    <key>type</key><string>error</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>10</integer>
// CHECK:    <key>col</key><integer>4</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:    <key>ranges</key>
// CHECK:    <array>
// CHECK:     <array>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>10</integer>
// CHECK:       <key>col</key><integer>6</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>10</integer>
// CHECK:       <key>col</key><integer>12</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:     </array>
// CHECK:    </array>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>

// DISABLE: mingw32
