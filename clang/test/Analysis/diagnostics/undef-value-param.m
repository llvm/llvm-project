// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx -analyzer-output=text -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx -analyzer-output=plist-multi-file %s -o - | FileCheck %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end
typedef const void * CFTypeRef;
extern void CFRelease(CFTypeRef cf);

@interface Cell : NSObject
- (void)test;
@end

@interface SpecialString
+ (id)alloc;
- (oneway void)release;
@end

typedef SpecialString* SCDynamicStoreRef;
static void CreateRef(SCDynamicStoreRef *storeRef, unsigned x);
SCDynamicStoreRef anotherCreateRef(unsigned *err, unsigned x);

@implementation Cell
- (void) test {
    SCDynamicStoreRef storeRef = 0; //expected-note{{Variable 'storeRef' initialized to nil}}
    CreateRef(&storeRef, 4); 
                             //expected-note@-1{{Calling 'CreateRef'}}
                             //expected-note@-2{{Returning from 'CreateRef'}}
    CFRelease(storeRef); //expected-warning {{Null pointer argument in call to CFRelease}}
                         //expected-note@-1{{Null pointer argument in call to CFRelease}}
}
@end

static void CreateRef(SCDynamicStoreRef *storeRef, unsigned x) {
    unsigned err = 0;
    SCDynamicStoreRef ref = anotherCreateRef(&err, x); // why this is being inlined?
    if (err) { 
               //expected-note@-1{{Assuming 'err' is not equal to 0}}
               //expected-note@-2{{Taking true branch}}
        CFRelease(ref);
        ref = 0;
    }
    *storeRef = ref;
}

//CHECK:  <dict>
//CHECK:   <key>files</key>
//CHECK:   <array>
//CHECK:   </array>
//CHECK:   <key>diagnostics</key>
//CHECK:   <array>
//CHECK:    <dict>
//CHECK:     <key>path</key>
//CHECK:     <array>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>33</integer>
//CHECK:        <key>col</key><integer>5</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>ranges</key>
//CHECK:       <array>
//CHECK:         <array>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>33</integer>
//CHECK:           <key>col</key><integer>5</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>33</integer>
//CHECK:           <key>col</key><integer>30</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:         </array>
//CHECK:       </array>
//CHECK:       <key>depth</key><integer>0</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Variable &apos;storeRef&apos; initialized to nil</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Variable &apos;storeRef&apos; initialized to nil</string>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>33</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>33</integer>
//CHECK:             <key>col</key><integer>21</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>34</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>34</integer>
//CHECK:             <key>col</key><integer>13</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>34</integer>
//CHECK:        <key>col</key><integer>5</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>ranges</key>
//CHECK:       <array>
//CHECK:         <array>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>34</integer>
//CHECK:           <key>col</key><integer>5</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>34</integer>
//CHECK:           <key>col</key><integer>27</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:         </array>
//CHECK:       </array>
//CHECK:       <key>depth</key><integer>0</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Calling &apos;CreateRef&apos;</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Calling &apos;CreateRef&apos;</string>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>42</integer>
//CHECK:        <key>col</key><integer>1</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>depth</key><integer>1</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Entered call from &apos;test&apos;</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Entered call from &apos;test&apos;</string>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>42</integer>
//CHECK:             <key>col</key><integer>1</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>42</integer>
//CHECK:             <key>col</key><integer>6</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>43</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>43</integer>
//CHECK:             <key>col</key><integer>12</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>43</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>43</integer>
//CHECK:             <key>col</key><integer>12</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>6</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>6</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>9</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>11</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>45</integer>
//CHECK:        <key>col</key><integer>9</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>ranges</key>
//CHECK:       <array>
//CHECK:         <array>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>45</integer>
//CHECK:           <key>col</key><integer>9</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>45</integer>
//CHECK:           <key>col</key><integer>11</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:         </array>
//CHECK:       </array>
//CHECK:       <key>depth</key><integer>1</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Assuming &apos;err&apos; is not equal to 0</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Assuming &apos;err&apos; is not equal to 0</string>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>9</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>45</integer>
//CHECK:             <key>col</key><integer>11</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>48</integer>
//CHECK:             <key>col</key><integer>9</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>48</integer>
//CHECK:             <key>col</key><integer>17</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>48</integer>
//CHECK:             <key>col</key><integer>9</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>48</integer>
//CHECK:             <key>col</key><integer>17</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>51</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>51</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>34</integer>
//CHECK:        <key>col</key><integer>5</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>ranges</key>
//CHECK:       <array>
//CHECK:         <array>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>34</integer>
//CHECK:           <key>col</key><integer>5</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>34</integer>
//CHECK:           <key>col</key><integer>27</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:         </array>
//CHECK:       </array>
//CHECK:       <key>depth</key><integer>1</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Returning from &apos;CreateRef&apos;</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Returning from &apos;CreateRef&apos;</string>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>control</string>
//CHECK:       <key>edges</key>
//CHECK:        <array>
//CHECK:         <dict>
//CHECK:          <key>start</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>34</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>34</integer>
//CHECK:             <key>col</key><integer>13</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:          <key>end</key>
//CHECK:           <array>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>37</integer>
//CHECK:             <key>col</key><integer>5</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:            <dict>
//CHECK:             <key>line</key><integer>37</integer>
//CHECK:             <key>col</key><integer>13</integer>
//CHECK:             <key>file</key><integer>0</integer>
//CHECK:            </dict>
//CHECK:           </array>
//CHECK:         </dict>
//CHECK:        </array>
//CHECK:      </dict>
//CHECK:      <dict>
//CHECK:       <key>kind</key><string>event</string>
//CHECK:       <key>location</key>
//CHECK:       <dict>
//CHECK:        <key>line</key><integer>37</integer>
//CHECK:        <key>col</key><integer>5</integer>
//CHECK:        <key>file</key><integer>0</integer>
//CHECK:       </dict>
//CHECK:       <key>ranges</key>
//CHECK:       <array>
//CHECK:         <array>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>37</integer>
//CHECK:           <key>col</key><integer>15</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:          <dict>
//CHECK:           <key>line</key><integer>37</integer>
//CHECK:           <key>col</key><integer>22</integer>
//CHECK:           <key>file</key><integer>0</integer>
//CHECK:          </dict>
//CHECK:         </array>
//CHECK:       </array>
//CHECK:       <key>depth</key><integer>0</integer>
//CHECK:       <key>extended_message</key>
//CHECK:       <string>Null pointer argument in call to CFRelease</string>
//CHECK:       <key>message</key>
//CHECK:  <string>Null pointer argument in call to CFRelease</string>
//CHECK:      </dict>
//CHECK:     </array>
//CHECK:     <key>description</key><string>Null pointer argument in call to CFRelease</string>
//CHECK:     <key>category</key><string>API Misuse (Apple)</string>
//CHECK:     <key>type</key><string>null passed to CFRetain/CFRelease/CFMakeCollectable</string>
//CHECK:    <key>issue_context_kind</key><string>Objective-C method</string>
//CHECK:    <key>issue_context</key><string>test</string>
//CHECK:    <key>issue_hash</key><integer>5</integer>
//CHECK:    <key>location</key>
//CHECK:    <dict>
//CHECK:     <key>line</key><integer>37</integer>
//CHECK:     <key>col</key><integer>5</integer>
//CHECK:     <key>file</key><integer>0</integer>
//CHECK:    </dict>
//CHECK:    </dict>
//CHECK:   </array>
//CHECK:  </dict>
//CHECK:  </plist>
