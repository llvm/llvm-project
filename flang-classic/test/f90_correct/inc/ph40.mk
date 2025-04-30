#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph40  ########


ph40: run
	

build:  $(SRC)/ph40.f
	-$(RM) ph40.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph40.f -o ph40.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph40.$(OBJX) check.$(OBJX) $(LIBS) -o ph40.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph40
	ph40.$(EXESUFFIX)

verify: ;

ph40.run: run

