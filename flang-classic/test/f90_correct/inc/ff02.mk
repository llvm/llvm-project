#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ff02  ########


ff02: run
	

build:  $(SRC)/ff02.f
	-$(RM) ff02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ff02.f -o ff02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ff02.$(OBJX) check.$(OBJX) $(LIBS) -o ff02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ff02
	ff02.$(EXESUFFIX)

verify: ;

ff02.run: run

