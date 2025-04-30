#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test be20  ########


be20: run
	

build:  $(SRC)/be20.f
	-$(RM) be20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/be20.f -o be20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) be20.$(OBJX) check.$(OBJX) $(LIBS) -o be20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test be20
	be20.$(EXESUFFIX)

verify: ;

be20.run: run

