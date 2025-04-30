#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka44  ########


ka44: run
	

build:  $(SRC)/ka44.f
	-$(RM) ka44.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka44.f -o ka44.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka44.$(OBJX) check.$(OBJX) $(LIBS) -o ka44.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka44
	ka44.$(EXESUFFIX)

verify: ;

ka44.run: run

