import lit.formats
import lit.TestRunner

# Custom format class for static analyzer tests
class AnalyzerTest(lit.formats.ShTest):

    def execute(self, test, litConfig):
        results = []

        # Parse any test requirements ('REQUIRES: ')
        saved_test = test
        lit.TestRunner.parseIntegratedTestScript(test)

        if 'z3' not in test.requires:
            results.append(self.executeWithAnalyzeSubstitution(
                saved_test, litConfig, '-analyzer-constraints=range'))

            if results[-1].code == lit.Test.FAIL:
                return results[-1]

        # If z3 backend available, add an additional run line for it
        if test.config.clang_staticanalyzer_z3 == '1':
            results.append(self.executeWithAnalyzeSubstitution(
                saved_test, litConfig, '-analyzer-constraints=z3 -DANALYZER_CM_Z3'))

        # Combine all result outputs into the last element
        for x in results:
            if x != results[-1]:
                results[-1].output = x.output + results[-1].output

        if results:
            return results[-1]
        return lit.Test.Result(lit.Test.UNSUPPORTED,
            "Test requires the following unavailable features: z3")

    def executeWithAnalyzeSubstitution(self, test, litConfig, substitution):
        saved_substitutions = list(test.config.substitutions)
        test.config.substitutions.append(('%analyze', substitution))
        result = lit.TestRunner.executeShTest(test, litConfig,
            self.execute_external)
        test.config.substitutions = saved_substitutions

        return result
