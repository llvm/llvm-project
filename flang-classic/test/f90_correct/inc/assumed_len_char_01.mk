#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test assumed_len_char_01  ########


assumed_len_char_01: run


build:  $(SRC)/assumed_len_char_01.f90
	-$(RM) assumed_len_char_01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/assumed_len_char_01.f90 -o assumed_len_char_01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) assumed_len_char_01.$(OBJX) check.$(OBJX) $(LIBS) -o assumed_len_char_01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test assumed_len_char_01
	assumed_len_char_01.$(EXESUFFIX)

verify: ;

assumed_len_char_01.run: run

