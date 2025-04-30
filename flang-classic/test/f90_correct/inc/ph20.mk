#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph20  ########


ph20: run
	

build:  $(SRC)/ph20.f
	-$(RM) ph20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph20.f -o ph20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph20.$(OBJX) check.$(OBJX) $(LIBS) -o ph20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph20
	ph20.$(EXESUFFIX)

verify: ;

ph20.run: run

