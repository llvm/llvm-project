#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph30  ########


ph30: run
	

build:  $(SRC)/ph30.f
	-$(RM) ph30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph30.f -o ph30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph30.$(OBJX) check.$(OBJX) $(LIBS) -o ph30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph30
	ph30.$(EXESUFFIX)

verify: ;

ph30.run: run

