#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test transfer function take quadruple precision  ########


qtransfer: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qtransfer.f08 fcheck.$(OBJX)
	-$(RM) qtransfer.$(EXESUFFIX) qtransfer.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qtransfer.f08 -o qtransfer.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qtransfer.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qtransfer.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qtransfer
	qtransfer.$(EXESUFFIX)

verify: ;

qtransfer.run: run

