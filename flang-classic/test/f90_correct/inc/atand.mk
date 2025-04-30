#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test atand  ########


atand: run


build:  $(SRC)/atand.f08
	-$(RM) atand.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/atand.f08 -o atand.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) atand.$(OBJX) check.$(OBJX) $(LIBS) -o atand.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test atand
	atand.$(EXESUFFIX)

verify: ;
