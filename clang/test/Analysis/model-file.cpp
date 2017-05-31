// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config faux-bodies=true,model-path=%S/Inputs/Models -analyzer-output=plist-multi-file -verify %s -o %t
// RUN: FileCheck --input-file=%t %s

typedef int* intptr;

// This function is modeled and the p pointer is dereferenced in the model
// function and there is no function definition available. The modeled
// function can use any types that are available in the original translation
// unit, for example intptr in this case.
void modeledFunction(intptr p);

// This function is modeled and returns true if the parameter is not zero
// and there is no function definition available.
bool notzero(int i);

// This functions is not modeled and there is no function definition.
// available
bool notzero_notmodeled(int i);

int main() {
  // There is a nullpointer dereference inside this function.
  modeledFunction(0);

  int p = 0;
  if (notzero(p)) {
   // It is known that p != 0 because of the information provided by the
   // model of the notzero function.
    int j = 5 / p;
  }

  if (notzero_notmodeled(p)) {
   // There is no information about the value of p, because
   // notzero_notmodeled is not modeled and the function definition
   // is not available.
    int j = 5 / p; // expected-warning {{Division by zero}}
  }

  return 0;
}

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
// CHECK-NEXT:            <key>line</key><integer>22</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>22</integer>
// CHECK-NEXT:            <key>col</key><integer>17</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>24</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>24</integer>
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
// CHECK-NEXT:       <key>line</key><integer>24</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>24</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>24</integer>
// CHECK-NEXT:          <key>col</key><integer>7</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>&apos;p&apos; initialized to 0</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>&apos;p&apos; initialized to 0</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>24</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>24</integer>
// CHECK-NEXT:            <key>col</key><integer>5</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>25</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>25</integer>
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
// CHECK-NEXT:            <key>line</key><integer>25</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>25</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
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
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>24</integer>
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
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>31</integer>
// CHECK-NEXT:            <key>col</key><integer>24</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>35</integer>
// CHECK-NEXT:            <key>col</key><integer>15</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>35</integer>
// CHECK-NEXT:            <key>col</key><integer>15</integer>
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
// CHECK-NEXT:       <key>line</key><integer>35</integer>
// CHECK-NEXT:       <key>col</key><integer>15</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>35</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>35</integer>
// CHECK-NEXT:          <key>col</key><integer>17</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Division by zero</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Division by zero</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Division by zero</string>
// CHECK-NEXT:    <key>category</key><string>Logic error</string>
// CHECK-NEXT:    <key>type</key><string>Division by zero</string>
// CHECK-NEXT:    <key>check_name</key><string>core.DivideZero</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>86cb845e4f1e6abde1c5b319d5b08eca</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>main</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>15</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>35</integer>
// CHECK-NEXT:    <key>col</key><integer>15</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  </array>
