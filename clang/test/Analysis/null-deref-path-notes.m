// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -analyzer-output=text -fblocks -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -analyzer-output=plist-multi-file -fblocks -Wno-objc-root-class %s -o %t
// RUN: FileCheck --input-file=%t %s

@interface Root {
@public
  int uniqueID;
}
- (id)initWithID:(int)newID;
- (void)refreshID;
@end

int testNull(Root *obj) {
  if (obj) return 0;
  // expected-note@-1 {{Assuming 'obj' is nil}}
  // expected-note@-2 {{Taking false branch}}

  int *x = &obj->uniqueID; // expected-note{{Variable 'x' initialized to a null pointer value}}
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}} expected-note{{Dereference of null pointer (loaded from variable 'x')}}
}


@interface Subclass : Root
@end

@implementation Subclass
- (id)initWithID:(int)newID {
  self = [super initWithID:newID]; // expected-note{{Value assigned to 'self'}}
  if (self) return self;
  // expected-note@-1 {{Assuming 'self' is nil}}
  // expected-note@-2 {{Taking false branch}}

  uniqueID = newID; // expected-warning{{Access to instance variable 'uniqueID' results in a dereference of a null pointer (loaded from variable 'self')}} expected-note{{Access to instance variable 'uniqueID' results in a dereference of a null pointer (loaded from variable 'self')}}
  return self;
}

@end


// CHECK:  <key>diagnostics</key>
// CHECK-NEXT:  <array>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>14</integer>
// CHECK-NEXT:       <key>col</key><integer>7</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>14</integer>
// CHECK-NEXT:          <key>col</key><integer>7</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>14</integer>
// CHECK-NEXT:          <key>col</key><integer>9</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Assuming &apos;obj&apos; is nil</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Assuming &apos;obj&apos; is nil</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>14</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>18</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>18</integer>
// CHECK-NEXT:            <key>col</key><integer>5</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>18</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>18</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>18</integer>
// CHECK-NEXT:          <key>col</key><integer>8</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Variable &apos;x&apos; initialized to a null pointer value</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Variable &apos;x&apos; initialized to a null pointer value</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>18</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>18</integer>
// CHECK-NEXT:            <key>col</key><integer>5</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>8</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>8</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>19</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>19</integer>
// CHECK-NEXT:       <key>col</key><integer>10</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>19</integer>
// CHECK-NEXT:          <key>col</key><integer>11</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>19</integer>
// CHECK-NEXT:          <key>col</key><integer>11</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Dereference of null pointer (loaded from variable &apos;x&apos;)</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Dereference of null pointer (loaded from variable &apos;x&apos;)</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;x&apos;)</string>
// CHECK-NEXT:    <key>category</key><string>Logic error</string>
// CHECK-NEXT:    <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>testNull</string>
// CHECK-NEXT:   <key>issue_hash</key><integer>6</integer>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>19</integer>
// CHECK-NEXT:    <key>col</key><integer>10</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>28</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>28</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>28</integer>
// CHECK-NEXT:          <key>col</key><integer>33</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value assigned to &apos;self&apos;</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value assigned to &apos;self&apos;</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>28</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>28</integer>
// CHECK-NEXT:            <key>col</key><integer>6</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>29</integer>
// CHECK-NEXT:       <key>col</key><integer>7</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>29</integer>
// CHECK-NEXT:          <key>col</key><integer>7</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>29</integer>
// CHECK-NEXT:          <key>col</key><integer>10</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Assuming &apos;self&apos; is nil</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Assuming &apos;self&apos; is nil</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>29</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>33</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>33</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>33</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>33</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>33</integer>
// CHECK-NEXT:          <key>col</key><integer>18</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Access to instance variable &apos;uniqueID&apos; results in a dereference of a null pointer (loaded from variable &apos;self&apos;)</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Access to instance variable &apos;uniqueID&apos; results in a dereference of a null pointer (loaded from variable &apos;self&apos;)</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Access to instance variable &apos;uniqueID&apos; results in a dereference of a null pointer (loaded from variable &apos;self&apos;)</string>
// CHECK-NEXT:    <key>category</key><string>Logic error</string>
// CHECK-NEXT:    <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>Objective-C method</string>
// CHECK-NEXT:   <key>issue_context</key><string>initWithID:</string>
// CHECK-NEXT:   <key>issue_hash</key><integer>6</integer>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>33</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  </array>
