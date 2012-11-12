//===---------------------AnsiTerminal.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//



#define ANSI_FG_COLOR_BLACK         30
#define ANSI_FG_COLOR_RED           31	
#define ANSI_FG_COLOR_GREEN         32	
#define ANSI_FG_COLOR_YELLOW        33	
#define ANSI_FG_COLOR_BLUE          34	
#define ANSI_FG_COLOR_PURPLE        35	
#define ANSI_FG_COLOR_CYAN          36	
#define ANSI_FG_COLOR_WHITE         37	

#define ANSI_BG_COLOR_BLACK         40
#define ANSI_BG_COLOR_RED           41	
#define ANSI_BG_COLOR_GREEN         42	
#define ANSI_BG_COLOR_YELLOW        44	
#define ANSI_BG_COLOR_BLUE          44	
#define ANSI_BG_COLOR_PURPLE        45	
#define ANSI_BG_COLOR_CYAN          46	
#define ANSI_BG_COLOR_WHITE         47	

#define ANSI_SPECIAL_FRAMED         51
#define ANSI_SPECIAL_ENCIRCLED      52

#define ANSI_CTRL_NORMAL            0
#define ANSI_CTRL_BOLD              1
#define ANSI_CTRL_FAINT             2
#define ANSI_CTRL_ITALIC            3
#define ANSI_CTRL_UNDERLINE         4
#define ANSI_CTRL_SLOW_BLINK        5
#define ANSI_CTRL_FAST_BLINK        6
#define ANSI_CTRL_IMAGE_NEGATIVE    7
#define ANSI_CTRL_CONCEAL           8
#define ANSI_CTRL_CROSSED_OUT       9

#define ANSI_ESC_START          "\033["
#define ANSI_ESC_END            "m"

#define ANSI_1_CTRL(ctrl1)          "\033["##ctrl1 ANSI_ESC_END
#define ANSI_2_CTRL(ctrl1,ctrl2)    "\033["##ctrl1";"##ctrl2 ANSI_ESC_END

namespace lldb_utility {

    namespace ansi {
        const char *k_escape_start	 = "\033[";
        const char *k_escape_end	 = "m";

        const char *k_fg_black	     = "30";
        const char *k_fg_red	     = "31";
        const char *k_fg_green	     = "32";
        const char *k_fg_yellow      = "33";
        const char *k_fg_blue	     = "34";
        const char *k_fg_purple      = "35";
        const char *k_fg_cyan        = "36";
        const char *k_fg_white	     = "37";

        const char *k_bg_black	     = "40";
        const char *k_bg_red	     = "41";
        const char *k_bg_green	     = "42";
        const char *k_bg_yellow      = "43";
        const char *k_bg_blue	     = "44";
        const char *k_bg_purple      = "45";
        const char *k_bg_cyan        = "46";
        const char *k_bg_white	     = "47";

        const char *k_ctrl_normal	     = "0";
        const char *k_ctrl_bold	         = "1";
        const char *k_ctrl_faint	     = "2";
        const char *k_ctrl_italic        = "3";
        const char *k_ctrl_underline	 = "4";
        const char *k_ctrl_slow_blink    = "5";
        const char *k_ctrl_fast_blink    = "6";
        const char *k_ctrl_negative	     = "7";
        const char *k_ctrl_conceal	     = "8";
        const char *k_ctrl_crossed_out	 = "9";

    }
}
