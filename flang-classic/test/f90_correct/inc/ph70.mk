#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ph70  ########


ph70: run
	

build:  $(SRC)/ph70.f
	-$(RM) ph70.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ph70.f -o ph70.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ph70.$(OBJX) check.$(OBJX) $(LIBS) -o ph70.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ph70
	ph70.$(EXESUFFIX)

verify: ;

ph70.run: run

