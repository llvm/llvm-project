#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test transpose function take quadruple precision  ########


qtranspose: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qtranspose.f08 fcheck.$(OBJX)
	-$(RM) qtranspose.$(EXESUFFIX) qtranspose.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qtranspose.f08 -o qtranspose.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qtranspose.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qtranspose.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qtranspose
	qtranspose.$(EXESUFFIX)

verify: ;

qtranspose.run: run

