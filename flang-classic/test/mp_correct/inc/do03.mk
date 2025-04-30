#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do03: do03.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do03.$(OBJX): $(SRC)/do03.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do03.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do03.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do03
run: ;
