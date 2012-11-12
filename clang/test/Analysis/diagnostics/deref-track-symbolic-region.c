// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-output=plist-multi-file %s -o - | FileCheck %s

struct S {
  int *x;
  int y;
};

int *foo();

void inlined(struct S *s, int m) {
  if (s->x)
    //expected-note@-1{{Taking false branch}}
    //expected-note@-2{{Assuming pointer value is null}}

    m++;

}
void test(struct S syz, int *pp) {
  int m = 0;
  syz.x = foo();
  inlined(&syz, m);
               // expected-note@-1{{Calling 'inlined'}}
               // expected-note@-2{{Returning from 'inlined'}}
  m += *syz.x; // expected-warning{{Dereference of null pointer (loaded from field 'x')}}
               // expected-note@-1{{Dereference of null pointer (loaded from field 'x')}}
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
//CHECK:      <key>kind</key><string>control</string>
//CHECK:      <key>edges</key>
//CHECK:       <array>
//CHECK:        <dict>
//CHECK:         <key>start</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>20</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>20</integer>
//CHECK:            <key>col</key><integer>5</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>22</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>22</integer>
//CHECK:            <key>col</key><integer>9</integer>
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
//CHECK:       <key>line</key><integer>22</integer>
//CHECK:       <key>col</key><integer>3</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>22</integer>
//CHECK:          <key>col</key><integer>3</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>22</integer>
//CHECK:          <key>col</key><integer>18</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>0</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Calling &apos;inlined&apos;</string>
//CHECK:      <key>message</key>
//CHECK: <string>Calling &apos;inlined&apos;</string>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>event</string>
//CHECK:      <key>location</key>
//CHECK:      <dict>
//CHECK:       <key>line</key><integer>11</integer>
//CHECK:       <key>col</key><integer>1</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>depth</key><integer>1</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Entered call from &apos;test&apos;</string>
//CHECK:      <key>message</key>
//CHECK: <string>Entered call from &apos;test&apos;</string>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>control</string>
//CHECK:      <key>edges</key>
//CHECK:       <array>
//CHECK:        <dict>
//CHECK:         <key>start</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>11</integer>
//CHECK:            <key>col</key><integer>1</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>11</integer>
//CHECK:            <key>col</key><integer>4</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>4</integer>
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
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>4</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>7</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>12</integer>
//CHECK:            <key>col</key><integer>7</integer>
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
//CHECK:       <key>line</key><integer>12</integer>
//CHECK:       <key>col</key><integer>7</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>12</integer>
//CHECK:          <key>col</key><integer>7</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>12</integer>
//CHECK:          <key>col</key><integer>10</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>1</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Assuming pointer value is null</string>
//CHECK:      <key>message</key>
//CHECK: <string>Assuming pointer value is null</string>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>event</string>
//CHECK:      <key>location</key>
//CHECK:      <dict>
//CHECK:       <key>line</key><integer>22</integer>
//CHECK:       <key>col</key><integer>3</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>22</integer>
//CHECK:          <key>col</key><integer>3</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>22</integer>
//CHECK:          <key>col</key><integer>18</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>1</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Returning from &apos;inlined&apos;</string>
//CHECK:      <key>message</key>
//CHECK: <string>Returning from &apos;inlined&apos;</string>
//CHECK:     </dict>
//CHECK:     <dict>
//CHECK:      <key>kind</key><string>control</string>
//CHECK:      <key>edges</key>
//CHECK:       <array>
//CHECK:        <dict>
//CHECK:         <key>start</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>22</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>22</integer>
//CHECK:            <key>col</key><integer>9</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>25</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>25</integer>
//CHECK:            <key>col</key><integer>3</integer>
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
//CHECK:            <key>line</key><integer>25</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>25</integer>
//CHECK:            <key>col</key><integer>3</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:          </array>
//CHECK:         <key>end</key>
//CHECK:          <array>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>25</integer>
//CHECK:            <key>col</key><integer>8</integer>
//CHECK:            <key>file</key><integer>0</integer>
//CHECK:           </dict>
//CHECK:           <dict>
//CHECK:            <key>line</key><integer>25</integer>
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
//CHECK:       <key>line</key><integer>25</integer>
//CHECK:       <key>col</key><integer>8</integer>
//CHECK:       <key>file</key><integer>0</integer>
//CHECK:      </dict>
//CHECK:      <key>ranges</key>
//CHECK:      <array>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>25</integer>
//CHECK:          <key>col</key><integer>13</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:         <dict>
//CHECK:          <key>line</key><integer>25</integer>
//CHECK:          <key>col</key><integer>13</integer>
//CHECK:          <key>file</key><integer>0</integer>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </array>
//CHECK:      <key>depth</key><integer>0</integer>
//CHECK:      <key>extended_message</key>
//CHECK:      <string>Dereference of null pointer (loaded from field &apos;x&apos;)</string>
//CHECK:      <key>message</key>
//CHECK: <string>Dereference of null pointer (loaded from field &apos;x&apos;)</string>
//CHECK:     </dict>
//CHECK:    </array>
//CHECK:    <key>description</key><string>Dereference of null pointer (loaded from field &apos;x&apos;)</string>
//CHECK:    <key>category</key><string>Logic error</string>
//CHECK:    <key>type</key><string>Dereference of null pointer</string>
//CHECK:   <key>issue_context_kind</key><string>function</string>
//CHECK:   <key>issue_context</key><string>test</string>
//CHECK:   <key>issue_hash</key><integer>6</integer>
//CHECK:   <key>location</key>
//CHECK:   <dict>
//CHECK:    <key>line</key><integer>25</integer>
//CHECK:    <key>col</key><integer>8</integer>
//CHECK:    <key>file</key><integer>0</integer>
//CHECK:   </dict>
//CHECK:   </dict>
//CHECK:  </array>
//CHECK: </dict>
//CHECK: </plist>
