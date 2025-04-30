#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gl00  ########


gl00: run
	

build:  $(SRC)/gl00.f
	-$(RM) gl00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gl00.f -o gl00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gl00.$(OBJX) check.$(OBJX) $(LIBS) -o gl00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gl00
	gl00.$(EXESUFFIX)

verify: ;

gl00.run: run

