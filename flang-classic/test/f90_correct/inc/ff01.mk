#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ff01  ########


ff01: run
	

build:  $(SRC)/ff01.f
	-$(RM) ff01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ff01.f -o ff01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ff01.$(OBJX) check.$(OBJX) $(LIBS) -o ff01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ff01
	ff01.$(EXESUFFIX)

verify: ;

ff01.run: run

