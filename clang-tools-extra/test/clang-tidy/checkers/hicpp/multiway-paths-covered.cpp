// RUN: %check_clang_tidy %s hicpp-multiway-paths-covered %t

enum OS { Mac,
          Windows,
          Linux };

struct Bitfields {
  unsigned UInt : 3;
  int SInt : 1;
};

int return_integer() { return 42; }

void bad_switch(int i) {
  switch (i) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch with only one case; use an if statement
  case 0:
    break;
  }
  // No default in this switch
  switch (i) {
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 0:
    break;
  case 1:
    break;
  case 2:
    break;
  }

  // degenerate, maybe even warning
  switch (i) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch statement without labels has no effect
  }

  switch (int j = return_integer()) {
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 0:
  case 1:
  case 2:
    break;
  }

  // Degenerated, only default case.
  switch (i) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: degenerated switch with default label only
  default:
    break;
  }

  // Degenerated, only one case label and default case -> Better as if-stmt.
  switch (i) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch could be better written as an if/else statement
  case 0:
    break;
  default:
    break;
  }

  unsigned long long BigNumber = 0;
  switch (BigNumber) {
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 0:
  case 1:
    break;
  }

  const int &IntRef = i;
  switch (IntRef) {
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 0:
  case 1:
    break;
  }

  char C = 'A';
  switch (C) {
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 'A':
    break;
  case 'B':
    break;
  }

  Bitfields Bf;
  // UInt has 3 bits size.
  switch (Bf.UInt) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: potential uncovered code path; add a default label
  case 0:
  case 1:
    break;
  }
  // All paths explicitly covered.
  switch (Bf.UInt) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
    break;
  }
  // SInt has 1 bit size, so this is somewhat degenerated.
  switch (Bf.SInt) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: switch with only one case; use an if statement
  case 0:
    break;
  }
  // All paths explicitly covered.
  switch (Bf.SInt) {
  case 0:
  case 1:
    break;
  }

  bool Flag = false;
  switch (Flag) {
    // CHECK-MESSAGES:[[@LINE-1]]:3: warning: switch with only one case; use an if statement
  case true:
    break;
  }

  switch (Flag) {
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: degenerated switch with default label only
  default:
    break;
  }

  // This `switch` will create a frontend warning from '-Wswitch-bool' but is
  // ok for this check.
  switch (Flag) {
  case true:
    break;
  case false:
    break;
  }
}

void unproblematic_switch(unsigned char c) {
  //
  switch (c) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
  case 16:
  case 17:
  case 18:
  case 19:
  case 20:
  case 21:
  case 22:
  case 23:
  case 24:
  case 25:
  case 26:
  case 27:
  case 28:
  case 29:
  case 30:
  case 31:
  case 32:
  case 33:
  case 34:
  case 35:
  case 36:
  case 37:
  case 38:
  case 39:
  case 40:
  case 41:
  case 42:
  case 43:
  case 44:
  case 45:
  case 46:
  case 47:
  case 48:
  case 49:
  case 50:
  case 51:
  case 52:
  case 53:
  case 54:
  case 55:
  case 56:
  case 57:
  case 58:
  case 59:
  case 60:
  case 61:
  case 62:
  case 63:
  case 64:
  case 65:
  case 66:
  case 67:
  case 68:
  case 69:
  case 70:
  case 71:
  case 72:
  case 73:
  case 74:
  case 75:
  case 76:
  case 77:
  case 78:
  case 79:
  case 80:
  case 81:
  case 82:
  case 83:
  case 84:
  case 85:
  case 86:
  case 87:
  case 88:
  case 89:
  case 90:
  case 91:
  case 92:
  case 93:
  case 94:
  case 95:
  case 96:
  case 97:
  case 98:
  case 99:
  case 100:
  case 101:
  case 102:
  case 103:
  case 104:
  case 105:
  case 106:
  case 107:
  case 108:
  case 109:
  case 110:
  case 111:
  case 112:
  case 113:
  case 114:
  case 115:
  case 116:
  case 117:
  case 118:
  case 119:
  case 120:
  case 121:
  case 122:
  case 123:
  case 124:
  case 125:
  case 126:
  case 127:
  case 128:
  case 129:
  case 130:
  case 131:
  case 132:
  case 133:
  case 134:
  case 135:
  case 136:
  case 137:
  case 138:
  case 139:
  case 140:
  case 141:
  case 142:
  case 143:
  case 144:
  case 145:
  case 146:
  case 147:
  case 148:
  case 149:
  case 150:
  case 151:
  case 152:
  case 153:
  case 154:
  case 155:
  case 156:
  case 157:
  case 158:
  case 159:
  case 160:
  case 161:
  case 162:
  case 163:
  case 164:
  case 165:
  case 166:
  case 167:
  case 168:
  case 169:
  case 170:
  case 171:
  case 172:
  case 173:
  case 174:
  case 175:
  case 176:
  case 177:
  case 178:
  case 179:
  case 180:
  case 181:
  case 182:
  case 183:
  case 184:
  case 185:
  case 186:
  case 187:
  case 188:
  case 189:
  case 190:
  case 191:
  case 192:
  case 193:
  case 194:
  case 195:
  case 196:
  case 197:
  case 198:
  case 199:
  case 200:
  case 201:
  case 202:
  case 203:
  case 204:
  case 205:
  case 206:
  case 207:
  case 208:
  case 209:
  case 210:
  case 211:
  case 212:
  case 213:
  case 214:
  case 215:
  case 216:
  case 217:
  case 218:
  case 219:
  case 220:
  case 221:
  case 222:
  case 223:
  case 224:
  case 225:
  case 226:
  case 227:
  case 228:
  case 229:
  case 230:
  case 231:
  case 232:
  case 233:
  case 234:
  case 235:
  case 236:
  case 237:
  case 238:
  case 239:
  case 240:
  case 241:
  case 242:
  case 243:
  case 244:
  case 245:
  case 246:
  case 247:
  case 248:
  case 249:
  case 250:
  case 251:
  case 252:
  case 253:
  case 254:
  case 255:
    break;
  }

  // Some paths are covered by the switch and a default case is present.
  switch (c) {
  case 1:
  case 2:
  case 3:
  default:
    break;
  }
}

OS return_enumerator() {
  return Linux;
}

// Enumpaths are already covered by a warning, this is just to ensure, that there is
// no interference or false positives.
// -Wswitch warns about uncovered enum paths and each here described case is already
// covered.
void switch_enums(OS os) {
  switch (os) {
  case Linux:
    break;
  }

  switch (OS another_os = return_enumerator()) {
  case Linux:
    break;
  }

  switch (os) {
  }
}

/// All of these cases will not emit a warning per default, but with explicit activation.
/// Covered in extra test file.
void problematic_if(int i, enum OS os) {
  if (i > 0) {
    return;
  } else if (i < 0) {
    return;
  }

  if (os == Mac) {
    return;
  } else if (os == Linux) {
    if (true) {
      return;
    } else if (false) {
      return;
    }
    return;
  } else {
    /* unreachable */
    if (true) // check if the parent would match here as well
      return;
  }

  // Ok, because all paths are covered
  if (i > 0) {
    return;
  } else if (i < 0) {
    return;
  } else {
    /* error, maybe precondition failed */
  }
}
