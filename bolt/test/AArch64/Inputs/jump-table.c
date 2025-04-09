volatile int g;
void switchy(int x) {
  switch (x) {
  case 0: g--; break;
  case 1: g++; break;
  case 2: g = 42; break;
  case 3: g += 17; break;
  case 4: g -= 66; break;
  case 5: g++; g--; break;
  case 6: g--; g++; break;
  case 66: g-=3; g++; break;
  case 8: g+=5; g--; break;
  case 10: g+=5; g--; break;
  case 12: g+=42; g--; break;
  case 15: g+=99; g--; break;
  case 20: switchy(g); break;
  case 21: g -= 1234; break;
  default: g = 0; break;
  }
}
