using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{

    public class ClangTidyProperties : DynamicPropertyComponent
    {
        private static ClangTidyProperties RootProperties_ = null;
        private CheckTree CheckTree_;
        private bool HasUnsavedChanges_ = false;

        public struct CheckMapping
        {
            public string CheckName;
            public string Property;
        }

        public ClangTidyProperties()
            : base(null)
        {
            AddClangCheckProperties();
            CheckTree_ = CheckTree.Build(this);
        }

        public ClangTidyProperties(DynamicPropertyComponent Parent)
            : base(Parent)
        {
            AddClangCheckProperties();
            CheckTree_ = CheckTree.Build(this);
        }

        static ClangTidyProperties()
        {
            RootProperties_ = new ClangTidyProperties(null);
        }

        public static ClangTidyProperties RootProperties
        {
            get { return RootProperties_; }
        }

        private void AddClangCheckProperties()
        {
            // Add each check in the check database
            HashSet<string> Categories = new HashSet<string>();
            foreach (var Check in CheckDatabase.Checks)
            {
                string Name = Check.Name.Replace('-', '_');
                List<Attribute> Attrs = new List<Attribute>();
                Attrs.Add(new CategoryAttribute(Check.Category));
                Attrs.Add(new DisplayNameAttribute(Check.Label));
                Attrs.Add(new DefaultValueAttribute(true));
                Attrs.Add(new DescriptionAttribute(Check.Desc));
                Attrs.Add(new ClangTidyCheckAttribute(Check.Name));
                Categories.Add(Check.Category);
                AddDynamicProperty<bool>(Check.Name, Attrs.ToArray());
            }

            // Add a category verb for each unique category.
            foreach (string Cat in Categories)
            {
                List<Attribute> Attrs = new List<Attribute>();
                Attrs.Add(new CategoryAttribute(Cat));
                Attrs.Add(new DisplayNameAttribute("(Category Verbs)"));
                Attrs.Add(new TypeConverterAttribute(typeof(CategoryVerbConverter)));
                Attrs.Add(new DefaultValueAttribute(CategoryVerb.None));
                AddDynamicProperty<CategoryVerb>(Cat + "Verb", Attrs.ToArray());
            }
        }

        public CheckTree GetCheckTree() { return CheckTree_; }
        public bool GetHasUnsavedChanges() { return HasUnsavedChanges_; }
        public void SetHasUnsavedChanges(bool Value) { HasUnsavedChanges_ = Value; }
    }
}
