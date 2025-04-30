#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test be00  ########


be00: run
	

build:  $(SRC)/be00.f
	-$(RM) be00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/be00.f -o be00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) be00.$(OBJX) check.$(OBJX) $(LIBS) -o be00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test be00
	be00.$(EXESUFFIX)

verify: ;

be00.run: run

