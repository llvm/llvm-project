//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

int main (int argc, char const *argv[])
{
    std::string stdstring("Hello\t\tWorld\nI am here\t\tto say hello\n"); //%self.addTearDownHook(lambda x: x.runCmd("setting set escape-non-printables true"))
    const char* constcharstar = stdstring.c_str();
    std::string longstring(
"I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
" is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
" a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
" for science, or something"
      "I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
      " is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
      " a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
      " for science, or something"
            "I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
            " is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
            " a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
            " for science, or something"
      );
    const char* longconstcharstar = longstring.c_str();
    return 0;     //% if self.TraceOn(): self.runCmd('frame variable')
      //% self.assertTrue(self.frame().FindVariable('stdstring').GetSummary() == '"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
     //% self.assertTrue(self.frame().FindVariable('constcharstar').GetSummary() == '"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
     //% self.runCmd("setting set escape-non-printables false")
     //% self.assertTrue(self.frame().FindVariable('stdstring').GetSummary() == '"Hello\t\tWorld\nI am here\t\tto say hello\n"')
     //% self.assertTrue(self.frame().FindVariable('constcharstar').GetSummary() == '"Hello\t\tWorld\nI am here\t\tto say hello\n"')
    //% self.assertTrue(self.frame().FindVariable('longstring').GetSummary().endswith('"...'))
    //% self.assertTrue(self.frame().FindVariable('longconstcharstar').GetSummary().endswith('"...'))
}

