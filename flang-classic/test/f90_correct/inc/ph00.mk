#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph00  ########


ph00: run
	

build:  $(SRC)/ph00.f
	-$(RM) ph00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph00.f -o ph00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph00.$(OBJX) check.$(OBJX) $(LIBS) -o ph00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph00
	ph00.$(EXESUFFIX)

verify: ;

ph00.run: run

